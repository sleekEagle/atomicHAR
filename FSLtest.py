from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from dataloaders import UTD_MHAD,PAMAP2,EMS
import matplotlib.pyplot as plt
from models import FCNN
import torch
from models.Model import AtomicHAR
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import logging
import random
import utils
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

#load the model
def load_model(conf,device):
    model_path=utils.get_model_path(conf)
    print(f'loading model from {model_path}...')

    athar_model=FCNN.HARmodel(conf,device)
    checkpoint = torch.load(model_path)
    athar_model.load_state_dict(checkpoint)
    athar_model.to(device)
    return athar_model

#store intermediate features from the network
def collect_features(dataloader,model,layer_name,device):
    model_layer = getattr(model, layer_name)
    # Register the hook to access the features
    model_layer.register_forward_hook(hook)
    features=torch.empty(0)
    for batch in dataloader:
        imu,activity_original,activity_remapped = batch
        _=model(imu.to(device))
        features=torch.cat((features,feature_output.detach().cpu()),dim=0)
    return features

#same as collect features but for a given input tensor
def get_features(model,input_data,layer_name,device):
    model_layer = getattr(model, layer_name)
    # Register the hook to access the features
    hook=model_layer.register_forward_hook(hook)
    _=model(input_data.to(device))
    f=feature_output
    hook.remove()
    return f


def hook(module, input, output):
    # This is where the output of the layer is stored during forward pass
    global feature_output
    feature_output = output

'''
replace the mean and std of BN layers
with the values from the FSL train data
in-place.
'''
def update_bn_stats(model,input_data,layers,device):
    input_data=input_data.to(device)
    #calculate the mean and std of the input data at different layers
    for fl_str,bl_str in layers:
        fl = getattr(model, fl_str)
        bl = getattr(model, bl_str)
        handle=fl.register_forward_hook(hook)
        
        _=model(input_data)
        mean=(feature_output.mean(dim=0)).detach()
        var=feature_output.var(dim=0).detach()
        bl.running_mean=mean
        bl.running_var=var
        handle.remove()
        
        # print(f'Updating the mean and std of BN layer {bl_str}')

def knn(model,train_inputs,train_l,test_inputs,test_l,device,n=5):
    _,train_f=model(train_inputs.to(device))
    train_f=train_f.cpu().detach().numpy()
    train_l=train_l.numpy()
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(train_f,train_l)
    _,test_f=model(test_inputs.to(device))
    test_f=test_f.cpu().detach().numpy()
    test_l=test_l.numpy()
    pred=knn.predict(test_f)
    correct=pred==test_l
    unique_l=np.unique(test_l)
    unique_l = np.sort(unique_l)
    correct_vals=[correct[test_l==l] for l in unique_l]
    class_acc=[np.sum(c)/len(c) for c in correct_vals]
    acc=np.sum(correct)/len(test_l)
    return acc,class_acc

def get_FSL_acc(conf,device,data,activity):
     #load weights
    athar_model=load_model(conf,device)
    if conf.FSL_test.finetune=='BN':
        #freeze all layers except the batch normalization layers
        for name, param in athar_model.named_parameters():
            if 'bn' not in name:
                param.requires_grad = False
        #get name of all free layers
        print('all free weights:')
        for name, param in athar_model.named_parameters():
            if param.requires_grad:
                print(name)

    # optimizer = torch.optim.Adam(athar_model.parameters(), lr=0.01)

    # Access the values of the batch normalization (bn) layer
    # bn_values = athar_model.blstm_bn.running_mean
    # print(bn_values)

    #register hook to acces features
    if conf.model.seq_model=='BLSTM':
        handle = athar_model.blstm_bn.register_forward_hook(hook)
    elif conf.model.seq_model=='transformer':
        print('this option is not implemented yet')
    elif conf.model.seq_model=='seq_CNN':
        handle = athar_model.cls_mp.register_forward_hook(hook)

     #*****************train the FSL classifier***************

    df=pd.DataFrame()
    df['activity']=activity.numpy()
    df['ind']=np.arange(len(activity))

    acc,n=0,0
    cls_acc=[]

    while True:
        if np.min(df.groupby('activity').size().values)<5:
            break
        n+=1
        #reload model 
        athar_model=load_model(conf,device)
        optimizer = optim.SGD(athar_model.parameters(), lr=0.001, momentum=0.9)
        
        train_df = df.groupby('activity').sample(n=5,random_state=1)
        train_indices=torch.tensor(train_df['ind'].values)
        train_inputs=data[train_indices]
        train_labels=activity[train_indices]   

        df=df.drop(train_df['ind'])
        test_indices=torch.tensor(df['ind'].values)
        test_input=data[test_indices]
        test_labels=activity[test_indices]  
        
        #define bn layers and respective feature layers
        layers=[['cnn12','bn1'],['cnn14','bn2'],['cnn16','bn3'],['blstm_lin','blstm_bn']]
        # print(athar_model.blstm_bn.running_mean)
        # print(athar_model.blstm_bn.running_var)
        original_cl=0
        if conf.FSL_test.finetune=='BN':
            for i in range(100):
                optimizer.zero_grad()
                pred,features=athar_model(train_inputs.to(device))
                center_loss=utils.center_loss(features,train_labels,device)
                if i==0:
                    original_cl=center_loss.item()
                center_loss.backward()
                optimizer.step()
            last_cl=center_loss.item()
            if last_cl>original_cl:
                print('using original model')
                athar_model=load_model(conf,device)

        acc_,class_acc_=knn(athar_model,train_inputs,train_labels,test_input,test_labels,device,n=5)
        print(f'running acc={acc_:.2f}')
        acc+=acc_
        cls_acc.append(class_acc_)
    print(f'FSL accuracy is {acc/n:.2f}  #runs={n}')
    cls_acc_mean=np.mean(np.array(cls_acc),axis=0)
    cls_acc_mean = [round(val, 2) for val in list(cls_acc_mean)]
    print(f'class accuracy={cls_acc_mean}')
    return acc/n

#plot the feature distributions of a given layer from two dataloaders
def plot_features(train_features,test_features,savepath):
    print('Plotting feature distributions...')
    fig, axs = plt.subplots(8, 8, figsize=(20, 20))
    for i in range(64):
        row = i // 8
        col = i % 8
        #get plot
        train_vals=train_features.view(-1,train_features.shape[1])[:,i].numpy()
        values,bins=np.histogram(train_vals, bins=100,density=True)
        bins_centers = 0.5*(bins[1:] + bins[:-1])
        axs[row, col].plot(bins_centers, values)
        
        test_vals=test_features.view(-1,test_features.shape[1])[:,i].numpy()
        values,bins=np.histogram(test_vals, bins=100,density=True)
        bins_centers = 0.5*(bins[1:] + bins[:-1])
        axs[row, col].plot(bins_centers, values,linestyle='dotted')

        # axs[row, col].set_title(f'Feature {i+1}')
    plt.tight_layout()
    plt.savefig(savepath, dpi=500)


def run_FSL(conf):
     # Check if the specified GPU is available
    if torch.cuda.is_available():
        n_gpus=torch.cuda.device_count()
        assert conf.gpu_index<n_gpus, f"The specified GPU index is not available. Available n GPUs: {n_gpus}"
        gpu_index=min(conf.gpu_index,n_gpus-1)
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")

    #load data
    dataset=conf.FSL_test.dataset
    if 'utdmhad' in dataset: 
        train_dataloader,test_dataloader=UTD_MHAD.get_dataloader(conf)
    if 'pamap2' in dataset: 
        train_dataloader,test_dataloader,fsl_dataloader=PAMAP2.get_dataloader(conf)
    if 'ems' in dataset:
        ems_data_loader=EMS.get_dataloader(conf)   

    athar_model=load_model(conf,device)

    if conf.FSL_test.type=='regular':
            #get test accuracy
        if conf.FSL_test.test_eval:
            eval_out=utils.eval(conf,athar_model,test_dataloader,device)
            print(f'test accuracy={eval_out:.2f}%')
        #collect features from dataloaders
        if 'pamap2' in conf.FSL_test.dataset:
            for batch in fsl_dataloader:
                pamap2_imu,activity_original,pamap2_activity_remapped = batch
                data=pamap2_imu
                activity=pamap2_activity_remapped
        if 'ems' in conf.FSL_test.dataset:
            for batch in ems_data_loader:
                ems_imu = batch
            #concatenate data from ems and pamap2
            ems_activity=torch.max(torch.unique(pamap2_activity_remapped)).item()+1
            ems_activity=(torch.ones(ems_imu.shape[0])*ems_activity).to(torch.int32)
            ems_imu_=ems_imu.swapaxes(1,2)
            data=torch.cat((pamap2_imu,ems_imu_),dim=0)
            activity=torch.cat((pamap2_activity_remapped,ems_activity),dim=0)

        acc=get_FSL_acc(conf,device,data,activity)

    elif conf.FSL_test.type=='fdist':
        layer=conf.FSL_test.layer
        train_features=collect_features(train_dataloader,athar_model,layer,device)
        test_features=collect_features(fsl_dataloader,athar_model,layer,device)
        plot_features(train_features,test_features,os.path.join(layer+'.png'))
    return acc

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf : DictConfig) -> None:
    run_FSL(conf)
   
if __name__ == "__main__":
    main()


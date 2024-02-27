from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from dataloaders import UTD_MHAD,PAMAP2
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

def hook(module, input, output):
    # This is where the output of the layer is stored during forward pass
    global feature_output
    feature_output = output


def get_FSL_acc(conf,device,test_dataloader,fsl_dataloader):
     #load weights
    model_path=utils.get_model_path(conf)
    print(f'loading model from {model_path}...')

    athar_model=FCNN.HARmodel(conf,device)
    checkpoint = torch.load(model_path)
    athar_model.load_state_dict(checkpoint)
    athar_model.to(device)

    #register hook to acces features
    if conf.model.seq_model=='BLSTM':
        handle = athar_model.blstm_bn.register_forward_hook(hook)
    elif conf.model.seq_model=='transformer':
        print('this option is not implemented yet')
    elif conf.model.seq_model=='seq_CNN':
        handle = athar_model.cls_cnn2.register_forward_hook(hook)

    #get test accuracy
    eval_out=utils.eval(conf,athar_model,test_dataloader,device)
    print(f'test accuracy={eval_out:.2f}%')

     #*****************train the FSL classifier***************
    #get fsl train data
    for batch in fsl_dataloader:
        imu,activity_original,activity_remapped = batch
    #get features
    outputs=athar_model(imu.to(device))
    train_features=feature_output.cpu().detach().numpy()
    if len(train_features.shape)==3:
        train_features=np.reshape(train_features,(train_features.shape[0],train_features.shape[1]*train_features.shape[2]))

    
    df=pd.DataFrame()
    df['activity']=activity_remapped.numpy()
    df['ind']=np.arange(len(activity_remapped))

    acc,n=0,0
    while True:
        if np.min(df.groupby('activity').size().values)<5:
            break
        n+=1
        train_df = df.groupby('activity').sample(n=5,random_state=1)
        data=train_features[train_df['ind'].values]
        labels=train_df['activity'].values
        #FSL classifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(data,labels)

        indices=train_df['ind'].values
        df=df.drop(train_df['ind'])
        data=train_features[df['ind'].values]
        labels=df['activity'].values

        pred=knn.predict(data)
        acc+=np.sum(pred==labels)/len(labels)
    print(f'FSL accuracy is {acc/n:.2f}  #runs={n}')
    return acc/n
    

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf : DictConfig) -> None:
    # Check if the specified GPU is available
    if torch.cuda.is_available():
        n_gpus=torch.cuda.device_count()
        assert conf.gpu_index<n_gpus, f"The specified GPU index is not available. Available n GPUs: {n_gpus}"
        gpu_index=min(conf.gpu_index,n_gpus-1)
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")

    #load data
    dataset=conf.data.dataset
    if dataset=='utdmhad': 
        train_dataloader,test_dataloader=UTD_MHAD.get_dataloader(conf)
    elif dataset=='pamap2': 
        _,test_dataloader,fsl_dataloader=PAMAP2.get_dataloader(conf)

    get_FSL_acc(conf,device,test_dataloader,fsl_dataloader)


if __name__ == "__main__":
    main()


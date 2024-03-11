import torch.nn as nn
import torch
import os
import numpy as np

def get_onehot(t,num_classes):
    return torch.squeeze(torch.eye(num_classes)[t.int()])

def get_files(path,ext='dat'):
    file_list = [file for file in os.listdir(path) if file.split('.')[-1]=='dat']
    return file_list

def stack_tensor(t,stack_dim=0):
    return torch.cat([t[i] for i in range(t.shape[0])],dim=stack_dim)

#apply stack tensor for each row in dimention 0
def batch_stack_tensor(t,stack_dim): 
    return torch.cat([torch.unsqueeze(stack_tensor(t[i],stack_dim=stack_dim),dim=0) for i in range(t.shape[0])],dim=0)

def unstack_tensor(t,n,len):
    return torch.cat([torch.unsqueeze(t[i*len:(i+1)*len],dim=0) for i in range(n)],dim=0)

def get_acc(gt,pred):
    label_pred=torch.argmax(pred,dim=1)
    label_gt=torch.argmax(gt,dim=1)
    acc=torch.sum(label_gt==label_pred).item()/gt.shape[0]
    acc=acc*100
    return acc

def eval(conf,model,dataloader,device):
    num_classes=len(conf.pamap2.train_ac)
    mean_acc=0
    for i,input in enumerate(dataloader):
        if conf.data.dataset=='pamap2':
            imu,_,activity=input
            activity_oh=get_onehot(activity,num_classes).to(device)        
        output,_=model(imu.to(device))
        acc=get_acc(output,activity_oh)
        mean_acc+=acc
    return mean_acc/len(dataloader)

def get_model_path(conf):
    model_path=conf.model.save_path
    dataset=conf.data.dataset
    lmd=conf.train.lmd
    if dataset=='pamap2':
        required_columns=conf.pamap2.required_columns
        dataset=conf.data.dataset
        cols= [col for col in conf[dataset].required_columns if (('hand' in col) or ('chest' in col) or ('ankle' in col))]
        train_s='trainG_'+','.join([item[-1:] for item in conf[dataset].train_subj])
        test_s='testG_'+','.join([item[-1:] for item in conf[dataset].test_subj])
        model_path=os.path.join(model_path,f'pamap2_{train_s}_{test_s}_nfeat_{len(cols)}_lmd{lmd}.pth')
    print('save model path:',model_path)
    return model_path

#get KL divergence between two data sequences
def get_kl_divergence(P_data,Q_data):
    min_val=min(np.min(P_data),np.min(Q_data))
    max_val=max(np.max(P_data),np.max(Q_data))
    vals,_=np.histogram(P_data,bins=100,range=(min_val,max_val))
    P_dist=vals/np.sum(vals)+1e-6
    vals,_=np.histogram(Q_data,bins=100,range=(min_val,max_val))
    Q_dist=vals/np.sum(vals)+1e-6
    kl_div=np.sum(P_dist*np.log(P_dist/Q_dist))
    return kl_div


#********************************************************************************
#********************************************************************************
#***************losses and metrics**********************************************
def dis_loss(features, labels,device):
    _,n=features.shape
    sums=(torch.zeros(labels.max().item() + 1,n).to(device).double()).index_add_(0, labels.to(device), features,alpha=1)
    nums=(torch.zeros(labels.max().item() + 1).to(device).double()).index_add_(0, labels.to(device), torch.ones_like(labels).to(device).double(),alpha=1)
    nums=nums.unsqueeze(-1).repeat(1,n)
    prototypes=sums/nums
    center_loss = torch.mean((features - prototypes[labels])**2)
    f_ind=torch.arange(0, features.shape[0])
    f_int_comb = torch.combinations(f_ind, r=2)
    inter_loss=-torch.mean((features[f_int_comb[:,0]]-features[f_int_comb[:,1]])**2)
    regulerize_loss = torch.mean((torch.norm(features, p=1,dim=1)-1)**2)**0.5
    dis_loss=center_loss
    return dis_loss
#********************************************************************************
#********************************************************************************


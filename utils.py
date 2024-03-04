import torch.nn as nn
import torch
import os

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


#********************************************************************************
#********************************************************************************
#***************losses and metrics**********************************************
def center_loss(features, labels,device):
    _,n=features.shape
    sums=(torch.zeros(labels.max().item() + 1,n).to(device).double()).index_add_(0, labels.to(device), features,alpha=1)
    nums=(torch.zeros(labels.max().item() + 1).to(device).double()).index_add_(0, labels.to(device), torch.ones_like(labels).to(device).double(),alpha=1)
    nums=nums.unsqueeze(-1).repeat(1,n)
    prototypes=sums/nums
    center_loss = torch.mean((features - prototypes[labels])**2)
    return center_loss
#********************************************************************************
#********************************************************************************


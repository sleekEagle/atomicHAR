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
import FSLtest
import wandb
# A logger for this file
log = logging.getLogger(__name__)


sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "fsl_acc"},
    "parameters": {
        "pamap2.model.atoms.hide_frac": {"values": [0.1,0.2,0.3]},
        "pamap2.model.atoms.num_indices": {"values": [1,2,3]},
        "pamap2.model.cnn.dropout": {"values": [[1,1,1],[1,1,0],[1,0,0],[0,0,0],[0,0,1],[0,1,1],[1,0,1],[0,1,0]]},
    },
}

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_imu_segments(imu,imu_last_seg,seg_len_list):
    bs,_,dim,_=imu.shape
    imu_segs_interp=torch.empty(0)
    # b_embeddings_list=torch.empty(0)
    for b in range(bs):            
        # b_seg_points=torch.cumsum(torch.tensor(seg_len_list[b]),dim=0)
        # b_embeddings=tr_out[(b_seg_points-1).long(),b,:]
        # b_embeddings_list=torch.cat((b_embeddings_list,b_embeddings),dim=0)
        #get imu segments that corresponds to the segments
        imu_segs=torch.split(imu[b,:int(imu_last_seg[b].item()),:,:],seg_len_list[b],dim=0)
        imu_comb_list=[]
        for i in range(len(imu_segs)):
            t=imu_segs[i]
            t_splt=torch.split(t,1,dim=0)
            imu_comb=torch.cat(t_splt,dim=2)[0]
            imu_comb_list.append(imu_comb)

        #resample the imu segments to a fixed size
        for attom in imu_comb_list:
            interp_seg=torch.nn.functional.interpolate(
                torch.unsqueeze(attom,dim=0),size=40)
            imu_segs_interp=torch.cat((imu_segs_interp,interp_seg),dim=0)  
    return imu_segs_interp

def plot_seg(imu,seg_len_list):
    b,seq,dim,l=imu.shape
    t=imu[0]
    t_ex=[item for item in t]
    d=torch.cat(t_ex,dim=1)
    d=torch.swapaxes(d,0,1)
    plt.plot(d)
    seg_ind=torch.tensor(seg_len_list[0])
    seg=torch.zeros(seq*l)
    seg[seg_ind*20]=1
    plt.plot(seg)
    wandb.log({"segmentation": plt})

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf : DictConfig) -> None:
    if conf.data.wandb:
        wandb.login()
        wandb_conf = OmegaConf.to_container(
            conf, resolve=True, throw_on_missing=True
        )
        wandb.init(project="atomicHAR-project",
                config=wandb_conf,
                )
    # conf=conf.params
    log.info('**********')
    log.info(conf)

    # Check if the specified GPU is available
    if torch.cuda.is_available():
        n_gpus=torch.cuda.device_count()
        assert conf.gpu_index<n_gpus, f"The specified GPU index is not available. Available n GPUs: {n_gpus}"
        gpu_index=min(conf.gpu_index,n_gpus-1)
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")
    
    dataset=conf.data.dataset
    global train_dataloader,test_dataloader,fsl_dataloader
    if not ('train_dataloader' in globals()):
        if dataset=='utdmhad': 
            train_dataloader,test_dataloader=UTD_MHAD.get_dataloader(conf)
        elif dataset=='pamap2': 
            train_dataloader,test_dataloader,fsl_dataloader=PAMAP2.get_dataloader(conf)
    print('dataloaders obtained...')

    num_classes=len(conf.pamap2.train_ac)
    athar_model=FCNN.HARmodel(conf,device)
    athar_model.to(device)
    MSE_loss_fn = nn.MSELoss()
    cls_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(athar_model.parameters(), lr=0.001)
    # optimizer = optim.SGD(athar_model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=100)
    model_path=utils.get_model_path(conf) 

    lr=get_lr(optimizer)
    log.info(f'new lr={lr}')

    min_loss=100
    plot_freq=100000
    for epoch in range(conf.model.epochs):
        mean_loss,mean_cls_loss,mean_cl,mean_acc=0,0,0,0
        if (epoch+1)%10==0:
            scheduler.step()
            lr=get_lr(optimizer)
            log.info(f'new lr={lr}')
        
        if (epoch+1)%plot_freq==0:
            plot=True
        else:
            plot=False
            plot_i=random.randint(0,len(train_dataloader))
        
        for i,input in enumerate(train_dataloader):
            if conf.data.dataset=='utdmhad': 
                imu,xyz,imu_mask,xyz_mask,imu_len,activity=input
            elif conf.data.dataset=='pamap2':
                imu,activity_original,activity=input
                activity_oh=utils.get_onehot(activity,num_classes).to(device)

            optimizer.zero_grad()

            output,features=athar_model(imu.to(device))
            cls_loss=cls_loss_fn(output,activity_oh)
            cl=0
            if conf.train.use_CL:
                cl=utils.center_loss(features,activity.to(device),device)
            loss=cls_loss+conf.train.lmd*cl
            acc=utils.get_acc(activity_oh,output)

            loss.backward()
            optimizer.step()
            mean_loss+=loss.item()
            mean_cls_loss+=cls_loss.item()
            mean_cl+=cl.item()
            mean_acc+=acc

        mean_loss=mean_loss/len(train_dataloader)
        mean_cls_loss=mean_cls_loss/len(train_dataloader)
        mean_cl=mean_cl/len(train_dataloader)
        mean_acc=mean_acc/len(train_dataloader)

        if mean_loss<min_loss:
            log.info('saving model...')
            min_loss=mean_loss
            torch.save(athar_model.state_dict(),model_path)

        log.info(f'Epoch={epoch}, cls loss = {mean_cls_loss:.5f},center loss = {mean_cl:.5f}, accuracy={mean_acc:.2f}')

        #***************eval*******************
        if epoch%10==0:
            eval_out=utils.eval(conf,athar_model,test_dataloader,device)
            log.info(f"test accuracy: {eval_out:.2f}")

    fsl_acc=FSLtest.run_FSL(conf)
    log.info(f'FSL accuracy is {fsl_acc:.2f}')
    if conf.data.wandb:
        wandb.log({'fsl_acc':fsl_acc})
    return fsl_acc

if __name__ == "__main__":
    main()

# sweep_id = wandb.sweep(sweep=sweep_configuration, project="atomic_sweep")
# wandb.agent(sweep_id, function=main, count=360)



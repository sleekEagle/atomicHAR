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
# A logger for this file
log = logging.getLogger(__name__)

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
        import wandb
        wandb.login()
        wandb_conf = OmegaConf.to_container(
            conf, resolve=True, throw_on_missing=True
        )
        wandb.init(project="atomicHAR",
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
    if dataset=='utdmhad': 
        train_dataloader,test_dataloader=UTD_MHAD.get_dataloader(conf)
    elif dataset=='pamap2': 
        train_dataloader,test_dataloader,_=PAMAP2.get_dataloader(conf)

    print('dataloaders obtained...')
    
    # athar_model=AtomicHAR(conf.pamap2.model,len(conf.utdmhad.train.actions))
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
        mean_loss,mean_imu_loss,mean_forcast_loss,mean_atom_loss,mean_cls_loss=0,0,0,0,0
        mean_acc=0

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

            output=athar_model(imu.to(device))

            cls_loss=cls_loss_fn(output,activity_oh)
            loss=cls_loss
            acc=utils.get_acc(activity_oh,output)

            loss.backward()
            optimizer.step()
            mean_loss+=loss.item()
            mean_imu_loss+=cls_loss.item()
            mean_acc+=acc
            # mean_forcast_loss+=forcast_loss.item()
            # mean_atom_loss+=atom_loss.item()
            # mean_cls_loss+=cls_loss.item()

        mean_loss=mean_loss/len(train_dataloader)
        mean_imu_loss=mean_imu_loss/len(train_dataloader)
        mean_forcast_loss=mean_forcast_loss/len(train_dataloader)
        mean_atom_loss=mean_atom_loss/len(train_dataloader)
        mean_cls_loss=mean_cls_loss/len(train_dataloader)
        mean_acc=mean_acc/len(train_dataloader)

        if mean_loss<min_loss:
            log.info('saving model...')
            min_loss=mean_loss
            torch.save(athar_model.state_dict(),model_path)

        # print(f'IMU loss = {mean_imu_loss:.5f},accuracy={acc:.2f}')
        if conf.data.wandb:
            wandb.log({"IMU_loss": mean_imu_loss,
                "forcast_loss": mean_forcast_loss,
                "cls_loss": mean_cls_loss,
                "atom loss":mean_atom_loss,
                'accuracy':acc})
        # plot_seg(imu,output['seg_len_list'])
        log.info(f'Epoch={epoch}, loss = {mean_imu_loss:.5f},accuracy={acc:.2f}')
        mean_loss=0
        mean_imu_loss=0
        mean_forcast_loss=0
        mean_atom_loss=0
        mean_acc=0

        #***************eval*******************
        if epoch%10==0:
            eval_out=utils.eval(conf,athar_model,test_dataloader,device)
            log.info(f"test accuracy: {eval_out:.2f}")

        #*************eval*******************
        # eval_out=utils.eval(athar_model,test_dataloader)
        # eval_imu_loss=eval_out['imu_loss']
        # eval_forcast_loss=eval_out['forcast_loss']
        # eval_atom_loss=eval_out['atom_loss']
        # print(f'Eval metrics: IMU loss = {eval_imu_loss:.5f},forcast loss= {eval_forcast_loss:.5f}, atom loss= {eval_atom_loss:.5f}')
        # wandb.log({"eval_IMU_loss": eval_imu_loss,
        #     "eval_forcast_loss": eval_forcast_loss,
        #     "eval_atom loss":eval_atom_loss})
        #************************************

        
    # real=torch.reshape(output['forcast_real'],(2,20,-1))
    # forcast=torch.reshape(output['forcast'],(2,20,-1))
    # d=(real-forcast)
    # d=torch.mean(d,dim=2)
    # d_plot=d[0,:].detach().cpu().numpy()



    # print(result.shape)
    # plt.plot(result.detach().cpu().numpy()[0,0,:])
    # plt.plot(imu[0,0,:].detach().cpu().numpy())
    if conf.data.wandb:
        wandb.log({'loss':min_loss})
    return min_loss

if __name__ == "__main__":
    main()



# output['imu_segs_interp']
# output['seg_len_list']
# b=0
# imu_=imu[b,:7,:,:]
# imu_=torch.reshape(imu_,(6,-1))
# imu_int=torch.nn.functional.interpolate(torch.unsqueeze(imu_,0),(20))[0]
# imu_segs=output['imu_segs_interp'][:2,:,:]
# imu_segs=torch.reshape(imu_segs,(6,40))
# imu_segs_int=torch.nn.functional.interpolate(torch.unsqueeze(imu_segs,0),(20))[0]


# plt.plot(imu_int[0,:].numpy())
# plt.plot(imu_segs_int[0,:].numpy())




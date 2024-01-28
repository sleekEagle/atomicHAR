from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from dataloaders import UTD_MHAD
import matplotlib.pyplot as plt
from models.Encoder import CNN
import torch
from models.Model import AtomicHAR
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import logging
# A logger for this file
log = logging.getLogger(__name__)
import wandb
wandb.login()

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
    wandb_conf = OmegaConf.to_container(
        conf, resolve=True, throw_on_missing=True
    )
    wandb.init(project="atomicHAR",
               config=wandb_conf,
               )

    # conf=conf.params
    log.info('**********')
    log.info(conf)
    train_dataloader,test_dataloader=UTD_MHAD.get_dataloader(conf)
    print('dataloaders obtained...')
    
    athar_model=AtomicHAR(conf.utdmhad.model)
    MSE_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(athar_model.parameters(), lr=0.001)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=100)


    lr=get_lr(optimizer)
    print(f'new lr={lr}')
    min_loss=100
    for epoch in range(conf.model.epochs):
        mean_loss,mean_imu_loss,mean_forcast_loss,mean_atom_loss=0,0,0,0
        print(f'epoch={epoch}')

        if (epoch+1)%10==0:
            print('reducing learning rate...')
            scheduler.step()
            lr=get_lr(optimizer)
            print(f'new lr={lr}')

        for i,input in enumerate(train_dataloader):
            imu,xyz,imu_mask,xyz_mask,imu_len=input

            optimizer.zero_grad()

            output=athar_model(imu,imu_mask,imu_len)
            # imu_segs_interp=get_imu_segments(imu,output['imu_last_seg'],output['seg_len_list'])
            
            imu_loss=MSE_loss_fn(imu*imu_mask,output['imu_gen']*imu_mask)
            forcast_loss=MSE_loss_fn(output['forcast_real']*output['forcast_mask'],
                                     output['forcast']*output['forcast_mask'])
            
            imu_atoms=output['imu_atoms']
            atom_loss=MSE_loss_fn(output['atom_gen']*output['atom_mask'],imu_atoms*output['atom_mask'])
            
            # xyz_loss=MSE_loss_fn(xyz*xyz_mask,xyz_gen*xyz_mask)
            loss=forcast_loss+imu_loss+atom_loss
            # print(f'IMU loss = {imu_loss:.5f},forcast loss= {forcast_loss:.5f}, atom loss= {atom_loss:.5f}, total loss={mean_loss:.2f}')

            loss.backward()
            optimizer.step()
            mean_loss+=loss.item()
            mean_imu_loss+=imu_loss.item()
            mean_forcast_loss+=forcast_loss.item()
            mean_atom_loss+=atom_loss.item()

        mean_loss=mean_loss/len(train_dataloader)
        mean_imu_loss=mean_imu_loss/len(train_dataloader)
        mean_forcast_loss=mean_forcast_loss/len(train_dataloader)
        mean_atom_loss=mean_atom_loss/len(train_dataloader)
        if mean_loss<min_loss:
            print('saving model...')
            min_loss=mean_loss
            torch.save(athar_model.state_dict(),conf.model.save_path)

        print(f'IMU loss = {mean_imu_loss:.5f},forcast loss= {mean_forcast_loss:.5f}, atom loss= {mean_atom_loss:.5f}, total loss={mean_loss:.2f}')
        wandb.log({"IMU_loss": mean_imu_loss,
            "forcast_loss": mean_forcast_loss,
            "atom loss":mean_atom_loss})
        # plot_seg(imu,output['seg_len_list'])
        log.info(f'IMU loss = {mean_imu_loss:.5f},forcast loss= {mean_forcast_loss:.5f}, atom loss= {mean_atom_loss:.5f}, total loss={mean_loss:.2f}')
        mean_loss=0
        mean_imu_loss=0
        mean_forcast_loss=0
        mean_atom_loss=0

        

    

    # real=torch.reshape(output['forcast_real'],(2,20,-1))
    # forcast=torch.reshape(output['forcast'],(2,20,-1))
    # d=(real-forcast)
    # d=torch.mean(d,dim=2)
    # d_plot=d[0,:].detach().cpu().numpy()



    # print(result.shape)
    # plt.plot(result.detach().cpu().numpy()[0,0,:])
    # plt.plot(imu[0,0,:].detach().cpu().numpy())
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




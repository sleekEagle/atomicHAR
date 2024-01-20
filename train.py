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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf : DictConfig) -> None:
    print(conf.utdmhad.path)
    train_dataloader,test_dataloader=UTD_MHAD.get_dataloader(conf)
    print('dataloaders obtained...')

    athar_model=AtomicHAR(conf.utdmhad.model)
    MSE_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(athar_model.parameters(), lr=0.001)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=100)


    lr=get_lr(optimizer)
    print(f'new lr={lr}')
    
    for epoch in range(1000):
        mean_loss,mean_imu_loss,mean_forcast_loss,mean_atom_loss=0,0,0,0
        print(f'epoch={epoch}')

        if (epoch+1)%10==0:
            print('reducing learning rate...')
            scheduler.step()
            lr=get_lr(optimizer)
            print(f'new lr={lr}')
        if(epoch==200):
            break

        for i,input in enumerate(train_dataloader):
            imu,xyz,imu_mask,xyz_mask,imu_len=input
            #create segmentation points
            # seg=torch.zeros_like(curvature)
            # seg_len=20
            # seg[:,0::seg_len]=1
            # seg[:,0]=0

            # split_imu=torch.split(imu,20,dim=2)
            # imu_chunks=torch.split(imu,20,dim=-1)
            # imu_chunks=torch.stack(imu_chunks)
            # imu_chunks=torch.swapaxes(imu_chunks,0,1)
            # bs,seq,dim,len=imu_chunks.shape
            # imu_chunks=imu_chunks.reshape(-1,dim,len)
            # imu_x_chunks=imu_chunks[:,0,:]

            # curvature=curvature/100.0
            # print(imu.shape)
            # result=cnn(imu)
            # print(result.shape)
            # src = torch.rand(10, 13, 512)
            # tr_out=transformer(src)

            optimizer.zero_grad()

            output=athar_model(imu,imu_mask,imu_len)
            
            imu_loss=MSE_loss_fn(imu*imu_mask,output['imu_gen']*imu_mask)
            forcast_loss=MSE_loss_fn(output['forcast_real']*output['forcast_mask'],
                                     output['forcast']*output['forcast_mask'])
            
            atom_loss=MSE_loss_fn(output['atom_gen'],output['imu_segs_interp'])
            
            # xyz_loss=MSE_loss_fn(xyz*xyz_mask,xyz_gen*xyz_mask)
            loss=imu_loss+forcast_loss+atom_loss
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

        print(f'****IMU loss = {mean_imu_loss:.5f},forcast loss= {mean_forcast_loss:.5f}, atom loss= {mean_atom_loss:.5f}, total loss={mean_loss:.2f}')
        mean_loss=0
        mean_imu_loss=0
        mean_forcast_loss=0
        mean_atom_loss=0

    print('broke')

    real=torch.reshape(output['forcast_real'],(2,20,-1))
    forcast=torch.reshape(output['forcast'],(2,20,-1))
    d=(real-forcast)
    d=torch.mean(d,dim=2)
    d_plot=d[0,:].detach().cpu().numpy()



    # print(result.shape)
    # plt.plot(result.detach().cpu().numpy()[0,0,:])
    # plt.plot(imu[0,0,:].detach().cpu().numpy())



if __name__ == "__main__":
    main()


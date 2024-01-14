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
        mean_loss=0
        print(f'epoch={epoch}')

        if (epoch+1)%10==0:
            print('reducing learning rate...')
            scheduler.step()
            lr=get_lr(optimizer)
            print(f'new lr={lr}')

        for i,input in enumerate(train_dataloader):
            imu,xyz=input
            bs,seq,dim,l=imu.shape
            imu_input=torch.reshape(imu,(-1,dim,l))

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

            result=athar_model(imu_input)

            loss=MSE_loss_fn(imu_input,result)
            loss.backward()
            optimizer.step()
            mean_loss+=loss.item()

        mean_loss=mean_loss/len(train_dataloader)
        print(f'loss={mean_loss}')
        mean_loss=0

    print('broke')
    print(result.shape)
    plt.plot(result.detach().cpu().numpy()[0,0,:])
    plt.plot(imu[0,0,:].detach().cpu().numpy())



if __name__ == "__main__":
    main()


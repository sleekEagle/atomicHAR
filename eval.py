from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from dataloaders import UTD_MHAD
import matplotlib.pyplot as plt
from models.Encoder import CNN
import torch
from models.Model import AtomicHAR
import torch.nn as nn

def eval(model,dataloader):
    MSE_loss_fn = nn.MSELoss()
    imu_loss_sum,forcast_loss_sum=0,0
    for i,input in enumerate(dataloader):
        imu,xyz,imu_mask,xyz_mask,imu_len=input
        output=model(imu,imu_mask,imu_len)
        imu_loss=MSE_loss_fn(imu*imu_mask,output['imu_gen']*imu_mask)
        forcast_loss=MSE_loss_fn(output['forcast_real']*output['forcast_mask'],
                                    output['forcast']*output['forcast_mask'])
        imu_loss_sum+=imu_loss.item()
        forcast_loss_sum+=forcast_loss.item()

        forcast_loss_seg=(torch.mean(torch.square(output['forcast_real']*output['forcast_mask']-output['forcast']*output['forcast_mask']),dim=-1))

        bs,seq,_,l=imu.shape
        imu_values=[]
        forc_loss=[]
        for b in range(bs):
            values=torch.cat([imu[b,j,:,:] for j in range(seq)],dim=-1)
            imu_values.append(values)
            batch_forcast_loss=forcast_loss_seg[b*seq:(b+1)*seq]
            batch_forcast_loss=torch.unsqueeze(batch_forcast_loss,dim=0).unsqueeze(dim=0)
            batch_forcast_loss=nn.functional.interpolate(batch_forcast_loss,seq*l)
            forc_loss.append(batch_forcast_loss[0][0])
        print('here')
    
    return imu_loss_sum/len(dataloader),forcast_loss/len(dataloader)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf : DictConfig) -> None:
    train_dataloader,test_dataloader=UTD_MHAD.get_dataloader(conf)
    print('dataloaders obtained...')
    athar_model=AtomicHAR(conf.utdmhad.model)
    # athar_model.load_state_dict(torch.load(conf.model.save_path))
    print('Waits loaded to the model...')
    imu_loss,forcast_loss=eval(athar_model,train_dataloader)
    # imu,xyz,imu_mask,xyz_mask,imu_len=next(iter(train_dataloader))
    # imu_data=imu[0,:,:,:]
    # imu_data_list=[item for item in imu_data]
    # imu_data=torch.cat(imu_data_list,dim=1)

    # output=athar_model(imu,imu_mask,imu_len)
    # print(output['forcast_loss'])


if __name__ == "__main__":
    main()

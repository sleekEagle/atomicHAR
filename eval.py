from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from dataloaders import UTD_MHAD
import matplotlib.pyplot as plt
from models.Encoder import CNN
import torch
from models.Model import AtomicHAR
import torch.nn as nn


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

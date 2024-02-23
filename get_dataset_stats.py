import hydra
from dataloaders import UTD_MHAD,PAMAP2
from omegaconf import DictConfig, OmegaConf
import torch
import matplotlib.pyplot as plt

#get stats for each data dim
@hydra.main(version_base=None, config_path="conf", config_name="config")
def get_PAMAP2_stats(conf : DictConfig) -> None:

    train_dataloader,test_dataloader,fsl_train_dataloader,fsl_test_dataloader=PAMAP2.get_dataloader(conf)
    imu_data=torch.empty(0)
    for j in range(10):
        print(f'epoch: {j}')
        for i,input in enumerate(train_dataloader):
            imu,activity=input
            imu_data=torch.cat((imu_data,imu),dim=0)

    max_values,_=torch.max(imu_data,dim=2)
    max_values,_=torch.max(max_values,dim=0)

    min_values,_=torch.min(imu_data,dim=2)
    min_values,_=torch.min(min_values,dim=0)

    print(f'min_vals: {list(min_values.numpy())}')
    print(f'max_vals: {list(max_values.numpy())}')

if __name__ == "__main__":
    get_PAMAP2_stats()
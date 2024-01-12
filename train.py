from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from dataloaders import UTD_MHAD
import matplotlib.pyplot as plt
from models.CNN import CNN
from models.Transformer import Transformer
import torch
from models.Model import AtomicHAR

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf : DictConfig) -> None:
    print(conf.utdmhad.path)
    train_dataloader,test_dataloader=UTD_MHAD.get_dataloader(conf)
    print('dataloaders obtained...')

    #create the CNN model
    cnn=CNN(conf)
    # cnn=cnn.double()

    transformer=Transformer(conf)
    # transformer=transformer.double()

    athar_model=AtomicHAR(conf)



    for i,input in enumerate(train_dataloader):
        imu,xyz_resampled,curvature,curv_class,seg=input
        print(imu.shape)
        # result=cnn(imu)
        # print(result.shape)
        # src = torch.rand(10, 13, 512)
        # tr_out=transformer(src)
        result=athar_model(imu)
        print(result.shape)


if __name__ == "__main__":
    main()


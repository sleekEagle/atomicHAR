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
    print(conf.utdmhad.path)
    train_dataloader,test_dataloader=UTD_MHAD.get_dataloader(conf)
    print('dataloaders obtained...')

    athar_model=AtomicHAR(conf.utdmhad.model)
    MSE_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(athar_model.parameters(), lr=0.001)
    for epoch in range(1000):
        if epoch==10:
            break
        for i,input in enumerate(train_dataloader):
            imu,xyz_resampled,curvature,curv_class,seg=input
            #create segmentation points
            # seg=torch.zeros_like(curvature)
            # seg_len=20
            # seg[:,0::seg_len]=1
            # seg[:,0]=0

            # split_imu=torch.split(imu,20,dim=2)
            xyz_chunks=torch.split(xyz_resampled,40,dim=-1)
            xyz_chunks=torch.stack(xyz_chunks)
            xyz_chunks=torch.swapaxes(xyz_chunks,0,1)
            bs,seq,dim,len=xyz_chunks.shape
            xyz_chunks=xyz_chunks.reshape(-1,dim,len)
            x_chunks=xyz_chunks[:,0,:]

            curvature=curvature/100.0
            # print(imu.shape)
            # result=cnn(imu)
            # print(result.shape)
            # src = torch.rand(10, 13, 512)
            # tr_out=transformer(src)
            optimizer.zero_grad()

            result=athar_model(imu)
            # result=torch.swapaxes(result,0,1)
            # result=torch.squeeze(result)

            loss=MSE_loss_fn(x_chunks,result)
            loss.backward()
            optimizer.step()

            print(loss.item())
    print('broke')


if __name__ == "__main__":
    main()


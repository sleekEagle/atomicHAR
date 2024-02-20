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

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf : DictConfig) -> None:
    #load data
    if conf.data.dataset=='utdmhad': 
        train_dataloader,test_dataloader=UTD_MHAD.get_dataloader(conf)
    elif conf.data.dataset=='pamap2': 
        _,test_dataloader,fsl_train_dataloader,fsl_test_dataloader=PAMAP2.get_dataloader(conf)

    #load weights
    athar_model=FCNN.HARmodel(conf.pamap2)
    checkpoint = torch.load(conf.model.save_path)
    athar_model.load_state_dict(checkpoint)

    #get test accuracy
    eval_out=utils.eval(conf,athar_model,test_dataloader)
    print(f'test accuracy={eval_out:.2f}%')

    #*****************train the FSL classifier***************
    #get fsl train data
    fsl_train_data = []
    for batch in fsl_train_dataloader:
        imu,activity = batch
        fsl_train_data.append((imu,activity))
    fsl_test_data = []
    for batch in fsl_test_dataloader:
        imu,activity = batch
        fsl_test_data.append((imu,activity))
    

if __name__ == "__main__":
    main()


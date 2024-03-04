import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

# Define your custom dataset class
class EMS(Dataset):
    def __init__(self, path,sample_freq=100,window_len_s=1):
        print('Getting data from ',path)
        participants = [os.path.join(path,d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith('P')]
        self.file_list = []
        for p in participants:
            session_dir = [os.path.join(p,d) for d in os.listdir(p) if os.path.isdir(os.path.join(p, d)) and d.startswith('s')]
            smartwatch_files=[os.path.join(s,'smartwatch','smartwatch_interp_100Hz.txt') for s in session_dir if os.path.exists(os.path.join(s, 'smartwatch'))]
            self.file_list.extend(smartwatch_files)
        self.min_vals= [-70.9484, -50.0299, -42.1422, -14.7317, -7.40136, -12.706]
        self.max_vals= [62.8596, 85.7169, 108.996, 18.4625, 10.0144, 10.5557]
        self.sample_freq=sample_freq
        self.window_len_s=window_len_s
        self.window_len=sample_freq*window_len_s

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file=self.file_list[index]
        data = np.genfromtxt(file, delimiter=' ')
        ts=data[:,0]
        accgyr=data[:,1:7]
        l,n=accgyr.shape
        if l>self.window_len:
            start_idx = np.random.randint(0, l-self.window_len)
            accgyr=accgyr[start_idx:start_idx+self.window_len]
            #mormalize
            min_vals=np.repeat(np.expand_dims(np.array(self.min_vals), axis=0), self.window_len, axis=0) 
            max_vals=np.repeat(np.expand_dims(np.array(self.max_vals), axis=0), self.window_len, axis=0)
            accgyr=(accgyr-min_vals)/(max_vals-min_vals)
        return accgyr

# @hydra.main(version_base=None, config_path="C:\\Users\\lahir\\code\\atomicHAR\\conf", config_name="config")
# def main(conf : DictConfig) -> None:
#     dataset = EMS(conf.EMS.path)
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#     # Iterate over the dataloader
#     d=torch.empty(0)
#     for batch in dataloader:
#         d=torch.cat((d,batch),dim=0)
#     print('here')

#     max_vals,_=d.max(dim=0)
#     max_vals,_=max_vals.max(dim=0)
#     min_vals,_=d.min(dim=0)
#     min_vals,_=min_vals.min(dim=0)



# if __name__ == "__main__":
#     main()

from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
import numpy as np
import hydra
import os
from torch.utils.data import Dataset

path='D:\\data\\UTD_MHAD\\'

def list_files(path,actions=[1,2,3],subjects=[1,2]):
    sk_directory=os.path.join(path,"Skeleton")
    in_directory=os.path.join(path,"Inertial")

    prefix=[]
    for a in actions:
        for s in subjects:
            prefix.append(f"a{a}_s{s}")

    sk_files = [d for d in os.listdir(sk_directory) if (os.path.splitext(d)[1]=='.mat' and 
                                                  '_'.join(d.split('_')[0:2]) in prefix)]
    sk_files.sort()
    in_files = [d for d in os.listdir(in_directory) if (os.path.splitext(d)[1]=='.mat' and 
                                                  '_'.join(d.split('_')[0:2]) in prefix)]
    in_files.sort()
    sk_files=[os.path.join(sk_directory,file) for file in sk_files]
    in_files=[os.path.join(in_directory,file) for file in in_files]
    return sk_files,in_files

def get_wrist_xyz(sk_file):
    wrist = loadmat(sk_file)['d_skel'][10]
    return wrist

def get_imu(in_file):
    mat = loadmat(os.path.join(in_file))
    imu_data=mat['d_iner']
    return imu_data

def resample_data(data,num_samples):
    t_axis=0 if data.shape[0]>10 else 1
    t=np.arange(data.shape[t_axis])
    if t_axis==0:
        x,y,z=data[:,0],data[:,1],data[:,2]
    elif t_axis==1:
        x,y,z=data[0,:],data[1,:],data[2,:]
    t_resample = np.linspace(0,t[-1], num_samples)
    x_resampled = np.interp(t_resample, t, x)
    y_resampled = np.interp(t_resample, t, y)
    z_resampled = np.interp(t_resample, t, z)
    return np.array([x_resampled,y_resampled,z_resampled])

#calculate curvature
def get_curvature(data):
    T=np.gradient(data,axis=1)
    # tmag=np.linalg.norm(T, axis=0)
    Tmag=np.repeat(np.expand_dims(np.linalg.norm(T, axis=0),0),3,axis=0)
    T_norm=T/Tmag
    dT=np.gradient(T_norm,axis=1)
    # dTmag=np.linalg.norm(dT, axis=0)
    k=dT/Tmag
    k_mag=np.linalg.norm(k, axis=0)
    return k_mag

# sk_files,in_files = list_files(path)

# sk_file=sk_files[0]
# in_file=in_files[0]
# xyz=get_wrist_xyz(sk_file)
# imu=get_imu(in_file)

# xyz_resampled=resample_data(xyz,imu.shape[0])
# curvature=get_curvature(xyz_resampled)
# plt.plot(curvature)
# plt.show()


class UTD_MHAD(Dataset):
    def __init__(self, data_dir,resample=True,curvature=True,actions=[1,2,3],subjects=[1,2]):
        self.data_dir=data_dir
        self.sk_files,self.in_files = list_files(data_dir,actions,subjects)
        self.resample=resample
        self.curvature=curvature

    def __len__(self):
        return len(self.sk_files)

    def __getitem__(self, idx):
        sk_file=self.sk_files[idx]
        in_file=self.in_files[idx]
        xyz=get_wrist_xyz(sk_file)
        imu=get_imu(in_file)
        if self.resample:
            xyz_resampled=resample_data(xyz,imu.shape[0])
        else:
            xyz_resampled=xyz
        if self.curvature:
            curvature=get_curvature(xyz_resampled)
        else:
            curvature=-1
        return imu,xyz_resampled,curvature
    

from torch.utils.data import DataLoader
training_data=UTD_MHAD(data_dir=path,actions=list(np.arange(1,22)),subjects=list(np.arange(1,9)))
train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)

cur_list=[]
for i,input in enumerate(train_dataloader):
    imu,xyz_resampled,curvature=input
    cur_list.extend(list(curvature[0].numpy()))

cur_ar=np.array(cur_list)
cur_ar=np.power(cur_ar)
# cur_ar=cur_ar[cur_ar<1000]
plt.hist(cur_list, bins=500, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of Values')
plt.grid(True)
plt.show()







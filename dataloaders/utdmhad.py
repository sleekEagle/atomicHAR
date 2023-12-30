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

#create curvature intervals
def get_curv_range(end=20000,base=1.2):
    ranges=[]
    i=0
    while i<end:
        range=pow(base,i)
        ranges.append([i,i+range])
        i=i+range
    return ranges

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
        self.ranges=get_curv_range()
        self.padded_len=400

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
            curv_class=np.zeros_like(curvature)
            for i,range in enumerate(self.ranges):
                idx=np.argwhere((curvature<range[1]) & (curvature>=range[0]))
                curv_class[idx]=i
        else:
            curvature=-1
            curv_class=-1

        #pad samples so there length (in time axis) would be self.padded_len
        curvature=np.expand_dims(curvature,axis=0)
        curv_class=np.expand_dims(curv_class,axis=0)
        imu=np.swapaxes(imu,0,1)
        imu_padded=self.get_padded_array(imu,self.padded_len)
        curvature_padded=self.get_padded_array(curvature,self.padded_len)
        curvature_padded=np.squeeze(curvature_padded)
        curv_class_padded=self.get_padded_array(curv_class,self.padded_len)
        curv_class_padded=np.squeeze(curv_class_padded)
        xyz_resampled_padded=self.get_padded_array(xyz_resampled,self.padded_len)

        return imu_padded,xyz_resampled_padded,curvature_padded,curv_class_padded
    
    def get_padded_array(self,array,padded_len):
        sample_len=array.shape[1]
        length_diff=padded_len-sample_len
        if length_diff > 0:
            padded_array = np.pad(array, ((0,0),(0,length_diff)), mode='constant', constant_values=0)
        else:
            padded_array = array
        return padded_array
    

from torch.utils.data import DataLoader
training_data=UTD_MHAD(data_dir=path,actions=list(np.arange(1,22)),subjects=list(np.arange(1,9)))
train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)

cur_list=[]
lens=[]
for i,input in enumerate(train_dataloader):
    imu,xyz_resampled,curvature,curv_class=input
    lens.append(curvature.shape[1])
    break
#     break
#     cur_list.extend(list(curvature[0].numpy()))

# cur_ar=np.array(cur_list)
# # cur_ar=np.power(cur_ar)
# # cur_ar=cur_ar[cur_ar<700]
# plt.hist(cur_ar, bins=500, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
# plt.xlabel('Values')
# plt.ylabel('Frequency')
# plt.title('Histogram of Values')
# plt.grid(True)
# plt.show()

ranges=get_curv_range()
nums=[]
cur_class=np.zeros_like(curvature)
for i,range in enumerate(ranges):
    idx=np.argwhere((curvature<range[1]) & (curvature>=range[0]))
    cur_class[idx]=i
    
    ar=cur_ar[cur_ar<range[1]]
    ar=ar[ar>range[0]]
    n=len(ar)
    nums.append(n)



# plt.plot(nums)
# plt.show()









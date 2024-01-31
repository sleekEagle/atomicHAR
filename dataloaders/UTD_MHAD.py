from scipy.io import loadmat
import os
import numpy as np
import os
from scipy import signal
from torch.utils.data import Dataset, DataLoader
import math

'''
how to create a dataloader

from torch.utils.data import DataLoader
training_data=UTD_MHAD(data_dir=path,actions=list(np.arange(1,22)),subjects=list(np.arange(1,9)))
train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
'''

path='C:\\Users\\lahir\\data\\UTD_MHAD\\'

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
    activities=[int(name.split('_')[0][1:])-1 for name in in_files]
    sk_files=[os.path.join(sk_directory,file) for file in sk_files]
    in_files=[os.path.join(in_directory,file) for file in in_files]
    return sk_files,in_files,activities

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
def get_curv_range(end=20000,base=2.5):
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
    '''
    imu_mask: [bs,seq_number,dim,items_in_seq]
            all items in a given sequence will be either valid or invalid
            i.e the mask is sequence level (sequence level resolution)
    '''
    def __init__(self, data_dir,
                 resample=True,
                 curvature=True,
                 imu_seg=True,
                 actions=[1,2,3],
                 subjects=[1,2]):
        self.data_dir=data_dir
        self.imu_seg=imu_seg
        self.sk_files,self.in_files,self.activities = list_files(data_dir,actions,subjects)
        uni_a=[str(item) for item in list(np.unique(self.activities))]
        text = f"{', '.join(uni_a)}"
        print('unique activities: '+text)
        self.max_ac=max(self.activities)
        self.resample=resample
        self.curvature=curvature
        self.ranges=get_curv_range()
        self.padded_len=400
        self.norms=[1.0,1.0,1.0,300.0,300.0,300.0]
        self.seq_len=20

    def __len__(self):
        return len(self.sk_files)

    def __getitem__(self, idx):
        sk_file=self.sk_files[idx]
        in_file=self.in_files[idx]
        xyz=get_wrist_xyz(sk_file)
        imu=get_imu(in_file)
        activity=self.activities[idx]
        #convert activity to one-hot format
        ac_onhot = np.zeros((self.max_ac+1), dtype=int)
        ac_onhot[activity]=1
        if self.resample:
            xyz_resampled=resample_data(xyz,imu.shape[0])
        else:
            xyz_resampled=xyz
        if self.curvature:
            curvature=get_curvature(xyz_resampled)
            curvature=signal.medfilt(curvature,3)
            curv_class=np.zeros_like(curvature)
            for i,range in enumerate(self.ranges):
                idx=np.argwhere((curvature<range[1]) & (curvature>=range[0]))
                curv_class[idx]=i
        else:
            curvature=-1
            curv_class=-1
        if self.imu_seg:
            #padding
            imu_len,dim=imu.shape
            imu=np.swapaxes(imu,0,1)
            imu_padded=self.get_padded_array(imu,self.padded_len)
            #break into segments
            imu_seg=np.split(imu_padded,self.seq_len,axis=1)
            imu_seg=np.stack(imu_seg)
            #normalize
            seq,dim,len=imu_seg.shape
            norm=np.array(self.norms)
            norm=np.expand_dims(norm,axis=1)
            norm=np.expand_dims(norm,axis=2)
            norm=np.swapaxes(norm,0,1)
            norm=np.repeat(norm,repeats=seq,axis=0)
            norm=np.repeat(norm,repeats=len,axis=2)
            imu_seg_norm=imu_seg/norm

            #pad xyz array
            xyz_padded=self.get_padded_array(xyz_resampled,self.padded_len)
            xyz_seg=np.split(xyz_padded,self.seq_len,axis=1)
            xyz_seg=np.stack(xyz_seg)
            xyz_first=xyz_seg[:,:,0]
            xyz_first=np.expand_dims(xyz_first,axis=2)
            xyz_first=np.repeat(xyz_first,repeats=len,axis=2)
            xyz_seg_norm=xyz_seg-xyz_first

            #create the mask 
            valid_data_seq=math.ceil(imu_len/self.seq_len)
            valid_data_seq=min(math.ceil(self.padded_len/self.seq_len),valid_data_seq)

            imu_mask=np.zeros_like(imu_seg_norm)
            imu_mask[0:valid_data_seq,:,:]=1

            xyz_mask=np.zeros_like(xyz_seg_norm)
            xyz_mask[0:valid_data_seq,:,:]=1

            return imu_seg_norm,xyz_seg_norm,imu_mask,xyz_mask,imu_len,ac_onhot
        
        else:
            seg=self.get_curve_segmentation(curvature)
            seg=np.expand_dims(seg,axis=0)
            seg_padded=self.get_padded_array(seg,self.padded_len)
            seg_padded=np.squeeze(seg_padded)
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

            return imu_padded,xyz_resampled_padded,curvature_padded,curv_class_padded,seg_padded
    
    def get_padded_array(self,array,padded_len):
        sample_len=array.shape[1]
        length_diff=padded_len-sample_len
        if length_diff > 0:
            padded_array = np.pad(array, ((0,0),(0,length_diff)), mode='constant', constant_values=0)
        else:
            padded_array = array
        return padded_array
    
    def get_curve_segmentation(self,c):
        grad=np.abs(np.gradient(c))
        grad_=(grad>5).astype(int)

        #get contiguous sections where one
        start_idx,end_idx=[],[]
        last_value=0
        for i,value in enumerate(grad_):
            if last_value==0 and value==1:
                start_idx.append(i)
            elif last_value==1 and value==0:
                end_idx.append(i-1)
            last_value=value
        #combine adjecent clusters if they are nearby
        thres1=10
        i=0
        l=max(len(start_idx),len(end_idx))
        last_end,last_end_idx=0,0
        while i<(l-1):
            if i<len(end_idx):
                idx=end_idx[i]
                last_end=idx
                last_end_idx=i
            if i+1>=len(start_idx):
                break
            next_start=start_idx[i+1]   
            diff=next_start-last_end  
            if diff < thres1:
                #remove start and end idx items from the list
                del start_idx[i+1]
                del end_idx[last_end_idx]  
            else:  
                i=i+1

        #remove clusters of 1 if they are too small
        thres2=7
        i=0
        while i<(len(end_idx)-1):
            if i>len(end_idx)-1:
                break
            idx=start_idx[i]
            next_start=end_idx[i]   
            diff=next_start-idx  
            if diff < thres2:
                #define a break point instead of the cluster
                mean_val=(start_idx[i]+end_idx[i])*0.5
                start_idx[i]=mean_val
                end_idx[i]=mean_val
            i=i+1
        #create the segmentation vector
        seg=np.zeros_like(c)
        for i,start in enumerate(start_idx):
            if i>0:
                seg[int(start)]=1
            if i < len(end_idx):
                end=end_idx[i]
                seg[int(end)]=1
        return seg
    
def get_dataloader(config):
    training_data=UTD_MHAD(data_dir=config.utdmhad.path,
                           imu_seg=config.utdmhad.imu_seg,
                           actions=list(config.utdmhad.train.actions),
                           subjects=list(config.utdmhad.train.subjects))
    train_dataloader = DataLoader(training_data, batch_size=config.utdmhad.train.bs, shuffle=True)

    test_data=UTD_MHAD(data_dir=path,
                       actions=list(config.utdmhad.test.actions),
                       subjects=list(config.utdmhad.test.subjects))
    test_dataloader = DataLoader(test_data, batch_size=config.utdmhad.test.bs, shuffle=True)
    
    return train_dataloader,test_dataloader


# import matplotlib.pyplot as plt
# cur_list=[]
# lens=[]
# for i,input in enumerate(train_dataloader):
#     imu,xyz_resampled,curvature,curv_class,seg=input
#     plt.plot(curvature[0]*0.001)  
#     plt.plot(seg[0])
#     plt.show()

#     lens.append(curvature.shape[1])
#     plt.figure()
#     plt.plot(curvature[0].numpy())
#     plt.show()
#     break

# def plot_3dpath(data):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(data[0,0,:],data[0,1,:],data[0,2,:],c='blue', marker='o')
#     ax.scatter(data[0,0,0],data[0,1,0],data[0,2,0],c='red', marker='o',s=200)
#     plt.show()

# from scipy import signal
# from scipy.ndimage import maximum_filter
# from scipy.ndimage import generic_filter

# def mode_filter(x):
#     unique, counts = np.unique(x, return_counts=True)
#     return unique[np.argmax(counts)]

# c=curvature[0].numpy()











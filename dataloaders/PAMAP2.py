import os
import sys
sys.path.append('C:\\Users\\lahir\\code\\atomicHAR')
import utils
import pandas as pd
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np

'''
adapted from 
https://www.kaggle.com/code/avrahamcalev/time-series-models-pamap2-dataset
'''


path='C:\\Users\\lahir\\data\\pamap2+physical+activity+monitoring\\PAMAP2_Dataset\\PAMAP2_Dataset\\'
dir='Optional'
required_columns=['time_stamp','activity_id',
              'hand_3D_acceleration_16_x','hand_3D_acceleration_16_y','hand_3D_acceleration_16_z',
              'hand_3D_gyroscope_x','hand_3D_gyroscope_y','hand_3D_gyroscope_z']

train_subj=[101,102,103,104,105,106,109]
test_subj=[107,108]
train_ac=[1,2,3,4,5,6,7,9,10,11,12,13,16,17,18,19,20,20]
test_ac=[1,2,3,4,5,6,7,9,10,11,12,13,16,17,18,19,20,20]


def load_activity_map():
    map = {}
    map[0] = 'transient'
    map[1] = 'lying'
    map[2] = 'sitting'
    map[3] = 'standing'
    map[4] = 'walking'
    map[5] = 'running'
    map[6] = 'cycling'
    map[7] = 'Nordic_walking'
    map[9] = 'watching_TV'
    map[10] = 'computer_work'
    map[11] = 'car driving'
    map[12] = 'ascending_stairs'
    map[13] = 'descending_stairs'
    map[16] = 'vacuum_cleaning'
    map[17] = 'ironing'
    map[18] = 'folding_laundry'
    map[19] = 'house_cleaning'
    map[20] = 'playing_soccer'
    map[24] = 'rope_jumping'
    return map

def generate_three_IMU(name):
    x = name +'_x'
    y = name +'_y'
    z = name +'_z'
    return [x,y,z]

def generate_four_IMU(name):
    x = name +'_x'
    y = name +'_y'
    z = name +'_z'
    w = name +'_w'
    return [x,y,z,w]

def generate_cols_IMU(name):
    # temp
    temp = name+'_temperature'
    output = [temp]
    # acceleration 16
    acceleration16 = name+'_3D_acceleration_16'
    acceleration16 = generate_three_IMU(acceleration16)
    output.extend(acceleration16)
    # acceleration 6
    acceleration6 = name+'_3D_acceleration_6'
    acceleration6 = generate_three_IMU(acceleration6)
    output.extend(acceleration6)
    # gyroscope
    gyroscope = name+'_3D_gyroscope'
    gyroscope = generate_three_IMU(gyroscope)
    output.extend(gyroscope)
    # magnometer
    magnometer = name+'_3D_magnetometer'
    magnometer = generate_three_IMU(magnometer)
    output.extend(magnometer)
    # oreintation
    oreintation = name+'_4D_orientation'
    oreintation = generate_four_IMU(oreintation)
    output.extend(oreintation)
    return output

def load_IMU():
    output = ['time_stamp','activity_id', 'heart_rate']
    hand = 'hand'
    hand = generate_cols_IMU(hand)
    output.extend(hand)
    chest = 'chest'
    chest = generate_cols_IMU(chest)
    output.extend(chest)
    ankle = 'ankle'
    ankle = generate_cols_IMU(ankle)
    output.extend(ankle)
    return output

def load_subjects(path,dir):
    files=utils.get_files(os.path.join(path,dir))
    output = pd.DataFrame()
    cols = load_IMU()

    for file in files:
        full_path=os.path.join(path,dir,file)
        subject = pd.read_table(full_path, header=None, sep='\s+')
        subject.columns = cols 
        subject=subject[required_columns]
        subject_cp=subject.copy()
        subj_id=int(file[-7:-4])
        subject_cp['id'] = subj_id
        #remove sections of transient activities. because we cannot guerentee that they are 
        # contiguous sections of activities
        condition=subject_cp['activity_id']==0
        subject_cp=subject_cp[~condition]

        #interpolate missing data
        subject_interp = subject_cp.interpolate(method='polynomial', order=2)

        output = pd.concat([output,subject_interp], ignore_index=True)
    output.reset_index(drop=True, inplace=True)
    return output

def get_data_split(data,subject_list,action_list):
    condition=data['activity_id'].isin(action_list) & data['id'].isin(subject_list)
    data_selected=data[condition]
    return data_selected

class PAMAP2(Dataset):
    '''
    imu_mask: [bs,seq_number,dim,items_in_seq]
            all items in a given sequence will be either valid or invalid
            i.e the mask is sequence level (sequence level resolution)
    '''
    def __init__(self,data,
                 actions=[1,2,3],
                 subjects=[1,2],
                 window_len=10):
        self.actions=actions
        self.subjects=subjects
        sample_freq=100
        self.n_samples=sample_freq*window_len

        self.data_split=get_data_split(data,subjects,actions)
        #get all participant and activity posibilities
        subj_list,ac_list=[],[]
        for s in self.subjects:
            # print(f'subject: {s}')
            condition=self.data_split['id']==s
            available_activities=list(self.data_split[condition]['activity_id'].unique())
            #check how many data samples are there for each activity
            n_samples_df=self.data_split[condition].groupby('activity_id').size()
            # print(n_samples_df)
            valid_activities=n_samples_df[n_samples_df>self.n_samples].index.to_numpy()
            # print(valid_activities)
            sub=[s]*len(valid_activities)
            subj_list.extend(sub)
            ac_list.extend(valid_activities) 
        self.subj_list=subj_list
        self.ac_list=ac_list

        #*************************************************************************
        #use these values to normalize
        self.min_vals= [-154.609, -94.6678, -118.846, -27.8044, -15.556, -14.2647]
        self.max_vals= [106.034, 157.611, 155.737, 26.4158, 16.9171, 16.548]
        #*************************************************************************


    def __len__(self):
        # return int(self.data_split.shape[0]/self.n_samples)
        return len(self.subj_list)

    def __getitem__(self, idx):
        condition=(self.data_split['id']==self.subj_list[idx]) & (self.data_split['activity_id']==self.ac_list[idx])
        data_selected=self.data_split[condition]
        start_idx=random.randint(0,data_selected.shape[0]-self.n_samples)
        data_sample=data_selected.iloc[start_idx:start_idx+self.n_samples]
        acc_x=data_sample['hand_3D_acceleration_16_x'].values
        acc_x=np.expand_dims(acc_x,axis=0)
        acc_y=data_sample['hand_3D_acceleration_16_y'].values
        acc_y=np.expand_dims(acc_y,axis=0)
        acc_z=data_sample['hand_3D_acceleration_16_z'].values
        acc_z=np.expand_dims(acc_z,axis=0)
        gyr_x=data_sample['hand_3D_gyroscope_x'].values
        gyr_x=np.expand_dims(gyr_x,axis=0)
        gyr_y=data_sample['hand_3D_gyroscope_y'].values
        gyr_y=np.expand_dims(gyr_y,axis=0)
        gyr_z=data_sample['hand_3D_gyroscope_z'].values
        gyr_z=np.expand_dims(gyr_z,axis=0)
        data_sample=np.concatenate((acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z),axis=0)

        activity=self.ac_list[idx]

        #normmalize data
        _,l=data_sample.shape
        max_vals=np.repeat(np.expand_dims(np.array(self.max_vals),axis=1),repeats=l,axis=1)
        min_vals=np.repeat(np.expand_dims(np.array(self.min_vals),axis=1),repeats=l,axis=1)
        data_sample=(data_sample-min_vals)/(max_vals-min_vals)
        return data_sample,activity

def get_dataloader(conf):
    optional_data = load_subjects(conf.pamap2.path,'Optional')
    protocol_data = load_subjects(conf.pamap2.path,'Protocol')
    data = pd.concat([optional_data,protocol_data], ignore_index=True) 

    training_data=PAMAP2(data,
                    actions=conf.pamap2.train_ac,
                    subjects=conf.pamap2.train_subj)
    train_dataloader = DataLoader(training_data, batch_size=conf.pamap2.train_bs, shuffle=True)

    test_data=PAMAP2(data,
                    actions=conf.pamap2.test_ac,
                    subjects=conf.pamap2.test_subj)
    test_dataloader = DataLoader(test_data, batch_size=conf.pamap2.test_bs, shuffle=True)
    
    return train_dataloader,test_dataloader

#get stats for each data dim
def get_stats():
    import torch
    import matplotlib.pyplot as plt

    train_dataloader,test_dataloader=get_dataloader(path)
    imu_data=torch.empty(0)
    for j in range(100):
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











































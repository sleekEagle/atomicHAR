import os
import sys
import utils
import pandas as pd
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

'''
adapted from 
https://www.kaggle.com/code/avrahamcalev/time-series-models-pamap2-dataset
'''

required_columns=['time_stamp','activity_id',
              'hand_3D_acceleration_16_x','hand_3D_acceleration_16_y','hand_3D_acceleration_16_z',
              'hand_3D_gyroscope_x','hand_3D_gyroscope_y','hand_3D_gyroscope_z',
              'hand_4D_orientation_x','hand_4D_orientation_y','hand_4D_orientation_z','hand_4D_orientation_w',
              'chest_3D_acceleration_16_x','chest_3D_acceleration_16_y','chest_3D_acceleration_16_z',
              'chest_3D_gyroscope_x','chest_3D_gyroscope_y','chest_3D_gyroscope_z',
              'chest_3D_magnetometer_x','chest_3D_magnetometer_y','chest_3D_magnetometer_z',
              'chest_4D_orientation_x','chest_4D_orientation_y','chest_4D_orientation_z','chest_4D_orientation_w',
              'ankle_3D_acceleration_16_x','ankle_3D_acceleration_16_y','ankle_3D_acceleration_16_z',
              'ankle_3D_gyroscope_x','ankle_3D_gyroscope_y','ankle_3D_gyroscope_z',
              'ankle_3D_magnetometer_x','ankle_3D_magnetometer_y','ankle_3D_magnetometer_z',
              'ankle_4D_orientation_x','ankle_4D_orientation_y','ankle_4D_orientation_z','ankle_4D_orientation_w']


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
                 window_len=1,
                 inv_overlap=0.5,normalize=True):
        self.actions=actions
        self.subjects=subjects
        self.window_len=window_len
        self.inv_overlap=inv_overlap
        self.normalize=normalize

        self.data_split=get_data_split(data,subjects,actions)
        self.data_split = self.data_split.reset_index(drop=True)

        #rename activities to vary from 0 to n
        self.new_actions=np.arange(len(actions))
        sub_ac_start_idx=[]
        actions=np.array(actions)
        for sub in subjects:
            for a in actions:
                print(f"subject: {sub}, activity: {a}")
                condition=(self.data_split['id']==sub) & (self.data_split['activity_id']==a)
                indices=self.data_split[condition].index.to_numpy()
                indices=indices[0:-self.window_len]
                start_idx=indices[::int(self.window_len*self.inv_overlap)]  
                for i in start_idx:
                    sub_ac_start_idx.append([sub,a,i])
        self.sub_ac_start_idx=sub_ac_start_idx

        # new_action=new_actions[np.where(actions==a)[0][0]]
        #get all participant and activity posibilities
        # subj_list,ac_list=[],[]
        # for s in self.subjects:
        #     # print(f'subject: {s}')
        #     condition=self.data_split['id']==s
        #     available_activities=list(self.data_split[condition]['activity'].unique())
        #     #check how many data samples are there for each activity
        #     n_samples_df=self.data_split[condition].groupby('activity').size()
        #     # print(n_samples_df)
        #     valid_activities=n_samples_df[n_samples_df>self.n_samples].index.to_numpy()
        #     # print(valid_activities)
        #     sub=[s]*len(valid_activities)
        #     subj_list.extend(sub)
        #     ac_list.extend(valid_activities) 
        # self.subj_list=subj_list
        # self.ac_list=ac_list

        #*************************************************************************
        #use these values to normalize
        self.min_vals= [-70.9484, -50.0299, -42.1422, -14.7317, -7.40136, -12.706, 6.02219e-07, -1.0108880737208994, -0.953427, -2.2584410232433645, -11.2318, -10.8517, -16.3059, -3.95072, -3.51333, -2.61079, -67.049, -85.8909, -48.8313, 3.77912e-06, -0.999467, -0.999635, -0.754197, -146.851, -97.1573, -158.926, -11.4788, -6.62893, -7.00578, -119.57, -93.5309, -87.7224, 2.79301e-06, -0.928419, -0.776711, -0.866813]
        self.max_vals= [62.8596, 85.7169, 108.996, 18.4625, 10.0144, 10.5557, 1.0000000000000002, 0.876677, 0.93495, 0.999455, 10.8285, 38.5496, 9.63652, 3.62787, 4.54031, 2.10232, 68.9376, 35.7563, 70.9868, 1.0000000000000002, 0.99927, 0.996864, 0.76637, 155.166, 156.19, 92.5958, 11.5109, 5.4311, 7.33777, 71.4946, 75.3057, 105.941, 1.0000000000000007, 1.0280537206471343, 0.924795, 0.9965292791359383]
        #*************************************************************************


    def __len__(self):
        # return int(self.data_split.shape[0]/self.n_samples)
        return len(self.sub_ac_start_idx)

    def __getitem__(self, idx):
        #select an activity and a subject
        subject=self.sub_ac_start_idx[idx][0]
        activity_original=self.sub_ac_start_idx[idx][1]
        start_idx=self.sub_ac_start_idx[idx][2]
        activity_remapped=self.new_actions[np.where(self.actions==activity_original)[0][0]]
        # condition=(self.data_split['id']==subject) & (self.data_split['activity']==activity)
        data_sample=self.data_split.iloc[start_idx:start_idx+self.window_len]
        if data_sample.shape[0]==0:
            print('here')

        data_vec_list=[]
        for colname in required_columns:
            if colname=='time_stamp' or colname=='activity_id':
                continue
            data_vec=data_sample[colname].values
            data_vec=np.expand_dims(data_vec,axis=0)
            data_vec_list.append(data_vec)
        data_sample=np.concatenate(data_vec_list,axis=0)

        # acc_x=data_sample['hand_3D_acceleration_16_x'].values
        # acc_x=np.expand_dims(acc_x,axis=0)
        # acc_y=data_sample['hand_3D_acceleration_16_y'].values
        # acc_y=np.expand_dims(acc_y,axis=0)
        # acc_z=data_sample['hand_3D_acceleration_16_z'].values
        # acc_z=np.expand_dims(acc_z,axis=0)
        # gyr_x=data_sample['hand_3D_gyroscope_x'].values
        # gyr_x=np.expand_dims(gyr_x,axis=0)
        # gyr_y=data_sample['hand_3D_gyroscope_y'].values
        # gyr_y=np.expand_dims(gyr_y,axis=0)
        # gyr_z=data_sample['hand_3D_gyroscope_z'].values
        # gyr_z=np.expand_dims(gyr_z,axis=0)
        # data_sample=np.concatenate((acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z),axis=0)

        # activity=self.data_split['activity'].iloc[idx]

        #normmalize data
        if self.normalize:
            _,l=data_sample.shape
            max_vals=np.repeat(np.expand_dims(np.array(self.max_vals),axis=1),repeats=l,axis=1)
            min_vals=np.repeat(np.expand_dims(np.array(self.min_vals),axis=1),repeats=l,axis=1)
            data_sample=(data_sample-min_vals)/(max_vals-min_vals)
        return data_sample,activity_original,activity_remapped

def get_dataloader(conf):
    df_list=[]
    if 'Optional' in conf.pamap2.data_types:
        optional_data = load_subjects(conf.pamap2.path,'Optional')
        df_list.append(optional_data)
    if 'Protocol' in conf.pamap2.data_types:
        protocol_data = load_subjects(conf.pamap2.path,'Protocol')
        df_list.append(protocol_data)
    data = pd.concat(df_list, ignore_index=True) 
    # data['activity']=pd.factorize(data['activity_id'])[0]
    
    window=conf.pamap2.window_len_s*conf.pamap2.sample_freq
    overlap=conf.pamap2.inv_overlap

    train_actions=conf.pamap2.train_ac
    division_type=conf.pamap2.division_type

    if division_type=='subject':
        train_subjects=conf.pamap2[conf.pamap2.train_subj]
        test_subjects=conf.pamap2[conf.pamap2.test_subj]
        training_data=PAMAP2(data,
                        actions=train_actions,
                        subjects=train_subjects, 
                        window_len=window, inv_overlap=overlap,normalize=conf.pamap2.normalize)
        train_dataloader = DataLoader(training_data, batch_size=conf.pamap2.train_bs, shuffle=True)

        test_data=PAMAP2(data,
                        actions=train_actions,
                        subjects=test_subjects,
                        window_len=window,inv_overlap=overlap,normalize=conf.pamap2.normalize)
        test_dataloader = DataLoader(test_data, batch_size=conf.pamap2.test_bs, shuffle=True)
        #FSL data
        fsl_subjects=conf.pamap2[conf.pamap2.FSL.test_subj]

    elif division_type=='regular':
        all_subj=conf.pamap2.group1+conf.pamap2.group2+conf.pamap2.group3
        reagular_data=PAMAP2(data,
                    actions=train_actions,
                    subjects=all_subj, 
                    window_len=window, inv_overlap=overlap,normalize=conf.pamap2.normalize)
        train_len=int(len(reagular_data)*conf.pamap2.split)
        test_len=len(reagular_data)-train_len
        train_dataloader,test_dataloader = torch.utils.data.random_split(reagular_data, [train_len, test_len])
        train_dataloader = DataLoader(train_dataloader, batch_size=conf.pamap2.train_bs, shuffle=True)
        test_dataloader = DataLoader(test_dataloader, batch_size=conf.pamap2.test_bs, shuffle=True)
        #use all subjects for few-shot learning
        fsl_subjects=all_subj
   
    #FSL data
    fsl_actions=conf.pamap2.FSL.test_ac
    fsl_overlap=conf.pamap2.FSL.inv_overlap

    fsl_data=PAMAP2(data,
                    actions=fsl_actions,
                    subjects=fsl_subjects,
                    window_len=window,inv_overlap=fsl_overlap,normalize=conf.pamap2.normalize)
    fsl_dataloader = DataLoader(fsl_data, batch_size=len(fsl_data), shuffle=True)
    
    return train_dataloader,test_dataloader,fsl_dataloader












































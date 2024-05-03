import torch
from torch.utils.data import DataLoader
import numpy as np
import csv
import pandas as pd
import os

def read_opportunity(datapath):
    files = {
        'training': [
            'S1-ADL1.dat','S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat',
            'S2-ADL1.dat', 'S2-ADL2.dat','S2-ADL5.dat', 'S2-Drill.dat',
            'S3-ADL1.dat', 'S3-ADL2.dat','S3-ADL5.dat', 'S3-Drill.dat', 
            'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat',
            'S1-ADL2.dat',  'S2-ADL3.dat', 'S2-ADL4.dat', 'S3-ADL3.dat', 'S3-ADL4.dat'

        ],
        'validation': [
        ],
        'test': [
        ]
    }

    label_map = [
        (0,      'Other'),
        (406516, 'Open Door 1'),
        (406517, 'Open Door 2'),
        (404516, 'Close Door 1'),
        (404517, 'Close Door 2'),
        (406520, 'Open Fridge'),
        (404520, 'Close Fridge'),
        (406505, 'Open Dishwasher'),
        (404505, 'Close Dishwasher'),
        (406519, 'Open Drawer 1'),
        (404519, 'Close Drawer 1'),
        (406511, 'Open Drawer 2'),
        (404511, 'Close Drawer 2'),
        (406508, 'Open Drawer 3'),
        (404508, 'Close Drawer 3'),
        (408512, 'Clean Table'),
        (407521, 'Drink from Cup'),
        (405506, 'Toggle Switch')
    ]
    label2id = {str(x[0]): i for i, x in enumerate(label_map)}
    id2label = [x[1] for x in label_map]

    cols = [
        38, 39, 40, 41, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59,
        64, 65, 66, 67, 68, 69, 70, 71, 72, 77, 78, 79, 80, 81, 82, 83, 84, 85,
        90, 91, 92, 93, 94, 95, 96, 97, 98, 103, 104, 105, 106, 107, 108, 109,
        110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
        125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 250]
    cols = [x-1 for x in cols] # labels for 18 activities (including other)

    data = {dataset: read_opp_files(datapath, files[dataset], cols, label2id,id2label)
            for dataset in ('training', 'validation', 'test')}

    return data, id2label

def read_opp_files(datapath, filelist, cols, label2id,id2label):
    data = []
    labels = []
    participants = []
    activity_name = []
    for i, filename in enumerate(filelist):
        participant=int(filename.split('/')[-1].split('-')[0][1:])
        nancnt = 0
        print('reading file %d of %d' % (i+1, len(filelist)))
        
        with open(os.path.join(datapath, 'dataset', filename), 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            for line in reader:
                elem = []
                for ind in cols:
                    elem.append(line[ind])
                # we can skip lines that contain NaNs, as they occur in blocks at the start
                # and end of the recordings.
                if sum([x == 'NaN' for x in elem]) == 0:
                    data.append([float(x) / 1000 for x in elem[:-1]])
                    labels.append(label2id[elem[-1]])
                    participants.append(participant)
                    activity_name.append(id2label[label2id[elem[-1]]])
    return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)+1,'participants' : np.asarray(participants),'name':np.asarray(activity_name)}


'''
mode: source or fsl
'''
class OPP(torch.utils.data.Dataset):
    def __init__(self, conf,mode):
        # Initialize your dataset here
        self.data, self.id2label = read_opportunity(conf.opp.path)
        df=pd.DataFrame()
        df['data']=list(self.data['training']['inputs'])
        df['label']=list(self.data['training']['targets'])
        df['participant']=list(self.data['training']['participants'])
        df['activity']=list(self.data['training']['name'])
        #select activities
        assert mode in ['source','fsl'], 'mode must be source or fsl'
        split=conf.opp.split
        if mode=='source':
            valid_acts=conf.opp[split].source_ac
            valid_subj=conf.opp.source_subj
        elif mode=='fsl':
            valid_acts=conf.opp[split].target_ac
            valid_subj=conf.opp.target_subj
        self.num_classes=len(valid_acts)
        df=df[df['activity'].isin(valid_acts)]
        df=df[df['participant'].isin(valid_subj)]
        #remap labels to start from 0 and be continuous
        df['label'] = pd.Categorical(df['label'])
        df['label'] = df['label'].cat.codes

        data_list,label_list,activity_name_list,participant_list=[],[],[],[]
        for i, g in df.groupby([(df['label'] != df['label'].shift()).cumsum()]):
            activity=g['label'].iloc[0]
            activity_name=g['activity'].iloc[0]
            participant=g['participant'].iloc[0]
            if((not(g.isnull().values.any()))):
                data_list.append(np.transpose(np.array([item for item in g['data'].values])))
                label_list.append(activity)
                participant_list.append(participant)
                activity_name_list.append(activity_name)
        
        window_len=int(conf.opp.sr*conf.opp.window_len_s)
        skip_num=int(window_len*conf.opp.overlap)
        self.window_data=[]
        self.window_label=[]
        self.window_participant=[]
        self.window_activity_name=[]
        for i in range(len(data_list)):
            data=data_list[i]
            label=label_list[i]
            participant=participant_list[i]
            activity_name=activity_name_list[i]
            for j in range(0,data.shape[1]-conf.opp.sr,skip_num):
                self.window_data.append(data[:,j:j+window_len])
                self.window_label.append(label)
                self.window_participant.append(participant)
                self.window_activity_name.append(activity_name)
        self.min_vals=[-1.426, -0.987, -0.105, -3.015, -2.563, -1.62, -0.627, -0.779, -0.885, -1.569, -1.321, -0.389, -4.413, -2.086, -3.484, -0.507, -0.859, -1.015, -1.822, -1.376, -0.78, -7.34, -2.944, -5.823, -0.807, -1.164, -1.094, -1.454, -1.031, -0.686, -3.051, -2.971, -2.48, -0.456, -0.849, -0.822, -1.8, -1.579, -1.115, -6.764, -4.818, -1.943, -0.702, -0.726, -0.843, -0.282, -0.087, -0.159, -5.137, -3.05, -4.74, -5.313, -2.028, -2.974, -8.561, -7.719, -7.2, -7.719, -6.388, -7.2, -0.267, -0.225, -0.091, -0.176, -6.623, -4.493, -4.842, -3.569, -2.035, -4.679, -10.799, -6.266, -8.035, -6.266, -6.442, -8.035, -0.227]
        self.max_vals=[0.216, 0.981, 1.263, 2.765, 2.271, 1.782, 0.733, 0.897, 0.591, -0.004, 0.912, 1.594, 3.871, 2.484, 2.556, 1.417, 0.941, 1.246, 0.638, 1.803, 1.559, 6.7, 4.338, 4.699, 1.528, 0.742, 2.686, -0.031, 1.052, 1.346, 3.549, 2.494, 2.293, 1.048, 0.853, 0.741, 0.652, 0.818, 1.268, 5.817, 4.178, 2.45, 1.384, 0.813, 1.059, 0.24, -0.01, 0.14, 2.285, 2.715, 2.353, 2.398, 5.573, 2.591, 6.388, 9.373, 8.888, 9.373, 8.561, 8.888, 0.318, 0.221, -0.006, 0.238, 2.95, 6.399, 2.229, 8.024, 5.497, 1.909, 6.442, 8.076, 8.909, 8.076, 10.799, 8.909, 0.233]
        self.normalize=conf.data.normalize
    
    def __getitem__(self, index):
        data=self.window_data[index]
        _,l=data.shape
        mins=np.repeat(np.expand_dims(np.array(self.min_vals),1),l,axis=1)
        maxs=np.repeat(np.expand_dims(np.array(self.max_vals),1),l,axis=1)
        if self.normalize:
            data=(data-mins)/(maxs-mins)
        return data,self.window_label[index],self.window_participant[index],self.window_activity_name[index]
        

    def __len__(self):
        return len(self.window_data)

'''
mode : source or fsl
'''
def get_dataloader(conf,mode):
    dataset = OPP(conf,mode)
    if mode=='source':
        #split the dataloader
        train_size = int(conf.opp.split_ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=conf.opp.train_bs, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=conf.opp.train_bs, shuffle=False)
        return train_dataloader,test_dataloader
    #create FSL dataloader
    elif mode=='fsl':
        fsl_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
        return fsl_dataloader

def get_stats(conf):
    train_dataloader,test_dataloader=get_dataloader(conf,mode='source')
    data=torch.empty(0)
    activity_list=torch.empty(0)
    print('getting stats...')
    for batch in train_dataloader:
        imu,activity,participant,ac_name = batch
        data=torch.cat((data,imu),dim=0)
        activity_list=torch.cat((activity_list,activity),dim=0)
    min_v,_=data.min(dim=0)
    min_v,_=min_v.min(dim=1)
    max_v,_=data.max(dim=0)
    max_v,_=max_v.max(dim=1)
    print(f'min_vals: {list(min_v.numpy())}')
    print(f'max_vals: {list(max_v.numpy())}')
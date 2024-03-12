import torch
from torch.utils.data import DataLoader
import numpy as np
import csv
import pandas as pd

def read_opportunity(datapath):
    files = {
        'training': [
            'S1-ADL1.dat',                'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat',
            'S2-ADL1.dat', 'S2-ADL2.dat',                               'S2-ADL5.dat', 'S2-Drill.dat',
            'S3-ADL1.dat', 'S3-ADL2.dat',                               'S3-ADL5.dat', 'S3-Drill.dat', 
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
        with open(datapath.rstrip('/') + '/dataset/%s' % filename, 'r') as f:
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


# Define your custom dataset class
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
        if mode=='source':
            valid_acts=conf.opp.source_ac
        elif mode=='fsl':
            valid_acts=conf.opp.target_ac
        df=df[df['activity'].isin(valid_acts)]
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
        
        window_len=int(conf.opp.sr*conf.opp.overlap)
        self.window_data=[]
        self.window_label=[]
        self.window_participant=[]
        self.window_activity_name=[]
        for i in range(len(data_list)):
            data=data_list[i]
            label=label_list[i]
            participant=participant_list[i]
            activity_name=activity_name_list[i]
            for j in range(0,data.shape[1]-conf.opp.sr,window_len):
                self.window_data.append(data[:,j:j+window_len])
                self.window_label.append(label)
                self.window_participant.append(participant)
                self.window_activity_name.append(activity_name)
        self.min_vals=[-1.93, -1.577, -1.394, -4.523, -4.281, -2.273, -0.929, -1.115, -1.246, -1.905, -1.92, -1.726, -12.75, -6.633, -6.735, -0.972, -2.789, -2.053, -1.879, -1.905, -1.876, -14.132, -9.206, -10.771, -1.52, -2.44, -3.013, -4.545, -5.59, -4.153, -14.469, -7.42, -9.635, -1.288, -1.274, -1.685, -5.643, -5.927, -5.111, -25.946, -11.983, -11.906, -3.367, -2.001, -1.537, -0.289, -0.093, -0.296, -9.59, -11.325, -10.663, -7.434, -7.094, -8.536, -37.139, -20.785, -13.114, -20.785, -17.312, -13.114, -0.303, -0.285, -0.093, -0.312, -10.269, -9.279, -9.324, -7.497, -10.534, -9.094, -27.941, -16.592, -15.677, -16.592, -23.075, -15.677, -0.342]
        self.max_vals=[1.254, 1.815, 1.641, 5.067, 3.716, 2.231, 1.709, 1.186, 1.449, 1.868, 1.813, 1.864, 14.054, 10.112, 5.245, 2.244, 1.597, 1.825, 1.894, 1.821, 1.911, 16.213, 10.385, 12.537, 2.11, 1.801, 2.944, 1.674, 5.688, 5.383, 14.177, 9.019, 6.975, 1.614, 1.454, 1.205, 4.787, 3.436, 4.629, 11.265, 9.964, 11.854, 2.195, 2.207, 2.133, 0.28, 0.07, 0.322, 9.168, 10.511, 7.722, 9.431, 7.849, 6.459, 17.312, 22.325, 13.363, 22.325, 37.139, 13.363, 0.318, 0.277, 0.085, 0.29, 8.518, 11.156, 13.358, 8.411, 8.755, 8.78, 23.075, 16.091, 22.72, 16.091, 27.941, 22.72, 0.273]
        self.normalize=conf.opp.normalize
    
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
    if mode=='source':
        dataset = OPP(conf,'fsl')
        #split the dataloader
        train_size = int(conf.opp.split * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=conf.opp.train_bs, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=conf.opp.train_bs, shuffle=False)
        return train_dataloader,test_dataloader

    #create FSL dataloader
    elif mode=='fsl':
        fsl_dataset=OPP(conf,'fsl')
        fsl_dataloader = DataLoader(fsl_dataset, batch_size=len(fsl_dataset), shuffle=True)
        return fsl_dataloader

def get_stats(conf):
    train_dataloader,test_dataloader=get_dataloader(conf)
    data=torch.empty(0)
    activity_list=torch.empty(0)
    print('getting stats...')
    for batch in test_dataloader:
        imu,activity,participant,ac_name = batch
        data=torch.cat((data,imu),dim=0)
        activity_list=torch.cat((activity_list,activity),dim=0)
    min_v,_=data.min(dim=0)
    min_v,_=min_v.min(dim=1)
    max_v,_=data.max(dim=0)
    max_v,_=max_v.max(dim=1)
    print(f'min_vals: {list(min_v.numpy())}')
    print(f'max_vals: {list(max_v.numpy())}')
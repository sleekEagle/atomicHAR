import torch
import torch.nn.functional as F
import utils
import math
from torch import nn, Tensor
from models.Transformer import TransformerModel
import torch

'''
how LSTMs work:
https://colah.github.io/posts/2015-08-Understanding-LSTMs/
https://www.youtube.com/watch?v=LfnrRPFhkuY
https://youtu.be/YCzL96nL7j0?si=agF25WbbCISxrajx
'''

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model=d_model
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.register_buffer('position', position)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: any tensor with positions as decimal numbers
        """
        atom_loc=torch.arange(x.shape[1])
        t_loc=torch.arange(torch.max(x)+1)
        atom_emb=self.pe[atom_loc,0,:]
        atom_emb = atom_emb.unsqueeze(0).unsqueeze(2).repeat(x.shape[0], 1, x.shape[2],1)
        t_emb=self.pe[t_loc,0,:][x,:]
        emb=torch.cat((atom_emb,t_emb),dim=3)
        # emb=emb.view(emb.shape[0],-1,emb.shape[-1])


        # atom_pos=self.position[atom_loc]
        # atom_pos = atom_pos.unsqueeze(0).unsqueeze(2).repeat(x.shape[0], 1, x.shape[2],1)
        # t_pos=self.position[t_loc][x,:]
        # emb_pos=torch.cat((atom_pos,t_pos),dim=3)
        # emb_pos_resh=emb_pos.view(-1,emb_pos.shape[-1])
        # emb_pos_unique=torch.unique(emb_pos_resh,dim=0)


        #remove duplicate embeddings
        # emb_resh=emb.view(-1,emb.shape[-1])
        # emb_unique=torch.unique(emb_resh,dim=0)
        return self.dropout(emb)

class AtomLayer(nn.Module):
    def __init__(self,num_indices,one_atm_per_time):
        super(AtomLayer, self).__init__()
        self.num_indices=num_indices
        self.threshold = nn.Parameter(torch.tensor(0.8))
        self.one_atm_per_time=one_atm_per_time

    def forward(self, x):
        threshold=self.threshold
        invalid_atoms=x<threshold
        x[invalid_atoms]=0
        if self.one_atm_per_time:
            vmax,imax=torch.max(x,dim=1)
            imax=imax.unsqueeze(1).repeat(1,x.shape[1],1)
            z_=torch.zeros_like(x)
            z_.scatter_(1,imax,1)
            x=x*z_
        sorted_tensor, indices = torch.sort(x, dim=2,descending=True)
        indices=indices[:,:,:self.num_indices]

        zeros=torch.zeros_like(x)
        ones=torch.ones_like(x)
        feat=zeros
        feat.scatter_(2, indices,ones)
        
        return feat,indices,torch.logical_not(invalid_atoms)

class HARmodel(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self,conf,device):
        super().__init__()
        self.device=device
        self.conf=conf
        # Extract features, 1D conv layers
        dataset=conf.data.dataset
        num_classes=len(conf[dataset].source_ac)

        #create cnn feature extractors
        self.feature_ext_list = nn.ModuleList()
        # cols= [col for col in conf[dataset].required_columns if (('hand' in col) or ('chest' in col) or ('ankle' in col))]
        # last_channels=len(cols)
        last_channels=conf[dataset].in_channels
        conc_channels=0
        for i in range(len(conf.model.feature_ext.layers)):
            module= nn.ModuleDict()
            c_=conf.model.feature_ext.layers[i]
            for j in range(len(c_)):
                c=c_[j]
                cnn_layer=nn.Conv1d(in_channels=last_channels,
                                    out_channels=c[0],
                                    kernel_size=c[1],stride=c[2]).double()
                
                last_channels=c[0]
                module[f'cnn_{i}_{j}']=cnn_layer
            conc_channels+=last_channels
            module[f'relu_{i}'] = nn.ReLU()
            bn=conf.model.feature_ext.bn[i]
            if bn:
                module[f'bn_{i}']=nn.BatchNorm1d(last_channels).double()
            d=conf.model.feature_ext.dropout[i]
            if d:
                module[f'drop_{i}'] = nn.Dropout(p=d)
            mp=conf.model.feature_ext.mp[i]
            module[f'mp_{i}'] = nn.MaxPool1d(kernel_size=mp[0],stride=mp[1],return_indices=False)

            self.feature_ext_list.append(module)
        #*******************
        self.seq_model=conf.model.seq_model.type
        if self.seq_model=='cnn':
            emb_conf=conf.model.seq_model.cnn.emb
            if conf.model.residual:
                ch_in=conc_channels
            else:
                ch_in=last_channels
            self.emb_cnn=nn.Conv1d(in_channels=ch_in, out_channels=emb_conf[0],
                               kernel_size=emb_conf[1],
                               stride=emb_conf[2]).double()
            # self.emb_bn=nn.BatchNorm1d(emb_conf[0]).double()
            cls_conf=conf.model.seq_model.cnn.cls
            emb_n_channles=emb_conf[0]
            self.cls_cnn=nn.Conv1d(in_channels=emb_n_channles, out_channels=num_classes,
                               kernel_size=cls_conf[1],
                               stride=cls_conf[2]).double()

        elif self.seq_model=='transformer':
            print('transformer model')
            self.pos_encoder=PositionalEncoding(int(self.seqconf.d_model*0.5)).double()
            self.transformer=TransformerModel(self.seqconf.d_model,self.seqconf.n_head,
                                            self.seqconf.dim_feedforward,self.seqconf.num_layers,
                                            num_classes,self.seqconf.dropout,self.device).double()
        elif self.seq_model=='BLSTM':
            blstmconf=conf[dataset].model.BLSTM
            print('BLSTM model')
            self.bilstm = nn.LSTM(192,
                                  blstmconf.hidden_size,
                                  blstmconf.num_layers,
                                  batch_first=True,
                                  bidirectional=True).double()
            self.blstm_lin = nn.Linear(blstmconf.hidden_size*2, blstmconf.dense).double()
            self.blstm_bn=nn.BatchNorm1d(blstmconf.dense).double()
            self.blstm_cls = nn.Linear(blstmconf.dense, self.num_classes).double()
        if conf.model.atoms.use_atoms:
            self.atom_layer_list=nn.ModuleList()
            self.num_indices=conf.model.atoms.num_indices
            self.atom_occuranes=conf.model.atoms.atm_occur
            self.atom_layer=AtomLayer(self.num_indices,conf.model.atoms.one_atm_per_time)
        self.hide_frac=conf.model.hide_frac

    def forward(self, x):
        #extract features
        features_list=[]
        for mod in self.feature_ext_list:
            for layer in mod:
                x=mod[layer](x)
            features_list.append(x)

        if self.conf.model.residual:
            features_int=[]
            _,_,l=features_list[-1].shape
            for i in range(len(features_list)-1):
                bs,n,_=features_list[i].shape
                features_int.append(F.interpolate(features_list[i].unsqueeze(0).unsqueeze(0),size=(bs, n, l),mode='trilinear', align_corners=False).squeeze())
            features_int.append(features_list[-1])
            features=torch.cat(features_int,dim=1)
        else:
            features=x
        
        if self.conf.model.atoms.use_atoms:
            features,_,_=self.atom_layer(features)

        #make random atoms zero 
        if self.hide_frac>0:
            bs,n_atoms,_=features.shape
            num_elements=int(n_atoms*self.hide_frac)
            ind=torch.arange(n_atoms).unsqueeze(0).repeat(bs,1).double()
            ind_sel=torch.multinomial(ind, num_samples=num_elements).unsqueeze(2).repeat(1,1,features.shape[2]).to(self.device)
            features.scatter_(1,ind_sel,0)

        if self.seq_model=='transformer':
            pos_enc=self.pos_encoder(features)
            valid_atoms=torch.gather(valid_atoms,2,features)
            valid_atoms=valid_atoms.unsqueeze(-1).repeat(1, 1, 1, pos_enc.shape[-1])
            pos_enc=pos_enc.view(pos_enc.shape[0],-1,pos_enc.shape[-1])
            valid_atoms=valid_atoms.view(valid_atoms.shape[0],-1,valid_atoms.shape[-1])

            pos_enc=pos_enc*valid_atoms.int()
            pos_enc=pos_enc.swapaxes(0,1)
            pred=F.softmax(self.transformer(pos_enc)[-1,:,:],dim=1)
        elif self.seq_model=='BLSTM':
            feat_conc=feat_conc.swapaxes(1,2)
            # print('BLSTM model')
            output, (hn, cn)=self.bilstm(feat_conc)
            last_out=output[:,-1,:]
            lin=self.blstm_lin(last_out)
            lin_bn=self.blstm_bn(lin)
            cls_features=self.blstm_cls(lin_bn)
            pred=F.softmax(cls_features,dim=1)
        elif self.seq_model=='cnn':
            emb_=self.emb_cnn(features)
            # emb_=self.emb_bn(emb_)
            emb=emb_.mean(dim=2)
            pred=F.softmax(self.cls_cnn(emb_).mean(dim=2),dim=1)
        return pred,emb
    


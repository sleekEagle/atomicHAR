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
    def __init__(self,num_indices):
        super(AtomLayer, self).__init__()
        self.num_indices=num_indices
        self.threshold = nn.Parameter(torch.tensor(0.8))

    def forward(self, x):
        threshold=self.threshold
        invalid_atoms=x<threshold
        x[invalid_atoms]=0
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
        # Extract features, 1D conv layers
        dataset=conf.data.dataset
        cnnconf=conf[dataset].model.cnn
        num_classes=len(conf[dataset].train_ac)
        #******block 1******
        self.cnn11=nn.Conv1d(in_channels=cnnconf.in_channels, out_channels=64, kernel_size=3,stride=1).double()
        self.cnn12=nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride=1).double()

        self.relu=nn.ReLU()
        self.bn=nn.BatchNorm1d(64).double()
        self.mp1=nn.MaxPool1d(kernel_size=3,stride=1,return_indices=False)
        self.dropout = nn.Dropout(p=0.2)
            
        self.cnn13=nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride=1).double()
        self.cnn14=nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2,stride=1).double()
        self.mp2=nn.MaxPool1d(kernel_size=3,stride=2,return_indices=False)

        self.cnn15=nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride=1).double()
        self.cnn16=nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2,stride=1).double()
        self.mp3=nn.MaxPool1d(kernel_size=3,stride=2,return_indices=False)
        #*******************

        self.embedding=nn.Conv1d(in_channels=cnnconf.num_atoms, out_channels=cnnconf.num_cls_features, kernel_size=2,stride=1).double()
        self.classifier=nn.Conv1d(in_channels=cnnconf.num_cls_features, out_channels=num_classes, kernel_size=3,stride=1).double()  
        self.num_classes=num_classes  

        self.seq_model=conf.model.seq_model
        self.seqconf=conf[dataset].model[self.seq_model]
        if self.seq_model=='transformer':
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

        self.num_indices=conf[dataset].model.atoms.num_indices
        self.atom_occuranes=conf[dataset].model.atoms.atm_occur
        self.atom_layer1=AtomLayer(self.num_indices)
        self.atom_layer2=AtomLayer(self.num_indices)
        self.atom_layer3=AtomLayer(self.num_indices)

        # self.lin_classifier=nn.Linear(256,num_classes).double()

    def forward(self, x):
        x=self.cnn11(x)
        x=self.cnn12(x)
        x=self.relu(self.bn(x))
        x=self.mp1(x)
        cnn1_out=x

        x=self.cnn13(x)
        x=self.cnn14(x)
        x=self.relu(self.bn(x))
        x=self.mp2(x)
        cnn2_out=x

        x=self.cnn15(x)
        x=self.cnn16(x)
        x=self.relu(self.bn(x))
        x=self.mp3(x)

        _,_,l=x.shape
        bs,n,_=cnn1_out.shape
        x1_resized = F.interpolate(cnn1_out.unsqueeze(0).unsqueeze(0), size=(bs, n, l), mode='trilinear', align_corners=False).squeeze()
        bs,n,_=cnn2_out.shape
        x2_resized = F.interpolate(cnn2_out.unsqueeze(0).unsqueeze(0), size=(bs, n, l), mode='trilinear', align_corners=False).squeeze()

        atoms1,indices1,valid_atoms1=self.atom_layer1(x1_resized)
        atoms2,indices2,valid_atoms2=self.atom_layer2(x2_resized)
        atoms3,indices3,valid_atoms3=self.atom_layer3(x)
        atoms=torch.cat((atoms1,atoms2,atoms3),dim=1)

        #make random atoms zero 
        num_elements = 5
        n_atoms=atoms.shape[1]
        ind=torch.arange(n_atoms).unsqueeze(0).repeat(bs,1).double()
        ind_sel=torch.multinomial(ind, num_samples=num_elements).unsqueeze(2).repeat(1,1,atoms.shape[2]).to(self.device)
        atoms.scatter_(1,ind_sel,0)

        # cnn16_weights = self.cnn16.weight
        # n,_,_=cnn16_weights.shape
        # cnn16_weights=cnn16_weights.view(n,-1)


        if self.seq_model=='transformer':
            pos_enc=self.pos_encoder(indices3)
            valid_atoms=torch.gather(valid_atoms,2,indices3)
            valid_atoms=valid_atoms.unsqueeze(-1).repeat(1, 1, 1, pos_enc.shape[-1])
            pos_enc=pos_enc.view(pos_enc.shape[0],-1,pos_enc.shape[-1])
            valid_atoms=valid_atoms.view(valid_atoms.shape[0],-1,valid_atoms.shape[-1])

            pos_enc=pos_enc*valid_atoms.int()
            pos_enc=pos_enc.swapaxes(0,1)
            pred=F.softmax(self.transformer(pos_enc)[-1,:,:],dim=1)
        elif self.seq_model=='BLSTM':
            atoms=atoms.swapaxes(1,2)
            # print('BLSTM model')
            output, (hn, cn)=self.bilstm(atoms)
            last_out=output[:,-1,:]
            lin=self.blstm_lin(last_out)
            lin_bn=self.blstm_bn(lin)
            pred=F.softmax(self.blstm_cls(lin_bn),dim=1)
        return pred

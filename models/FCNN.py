import torch
import torch.nn.functional as F
import utils
import math
from torch import nn, Tensor
from models.Transformer import TransformerModel
import torch

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



class HARmodel(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self,conf,device):
        super().__init__()

        # Extract features, 1D conv layers
        cnnconf=conf.model.cnn
        num_classes=len(conf.train_ac)
        self.cnn1=nn.Conv1d(in_channels=cnnconf.in_channels, out_channels=cnnconf.out_channels, kernel_size=cnnconf.kernel_size,stride=1).double()
        self.cnn2=nn.Conv1d(in_channels=cnnconf.out_channels, out_channels=cnnconf.out_channels, kernel_size=cnnconf.kernel_size,stride=1).double()

        self.relu=nn.ReLU()
        self.bn=nn.BatchNorm1d(64).double()
        self.mp=nn.MaxPool1d(kernel_size=5,stride=2,return_indices=False)
        self.mp_atom=nn.MaxPool1d(kernel_size=5,stride=4,return_indices=True)
            
        self.cnn3=nn.Conv1d(in_channels=cnnconf.out_channels, out_channels=cnnconf.out_channels, kernel_size=cnnconf.kernel_size,stride=1).double()
        self.cnn4=nn.Conv1d(in_channels=cnnconf.out_channels, out_channels=cnnconf.num_atoms, kernel_size=2,stride=1).double()

        self.embedding=nn.Conv1d(in_channels=cnnconf.num_atoms, out_channels=cnnconf.num_cls_features, kernel_size=2,stride=1).double()
        self.classifier=nn.Conv1d(in_channels=cnnconf.num_cls_features, out_channels=num_classes, kernel_size=3,stride=1).double()  
        self.num_classes=num_classes  

        trnsconf=conf.model.transformer
        self.pos_encoder=PositionalEncoding(int(trnsconf.d_model*0.5)).double()
        self.device=device
        self.transformer=TransformerModel(trnsconf.d_model,trnsconf.n_head,
                                          trnsconf.dim_feedforward,trnsconf.num_layers,
                                          num_classes,trnsconf.dropout,self.device).double()
        self.num_indices=trnsconf.num_indices
        self.atom_occuranes=trnsconf.atm_occur
        # self.lin_classifier=nn.Linear(256,num_classes).double()

    def forward(self, x):
        x=self.cnn1(x)
        x=self.cnn2(x)
        x=self.relu(self.bn(x))
        # atom_args=torch.argwhere(x>1.01)
        # diff_t=torch.diff(atom_args,dim=1)
        x=self.mp(x)

        x=self.cnn3(x)
        x=self.cnn4(x)
        x=self.relu(self.bn(x))
        x_mp=self.mp(x)

        atom_x,ind=self.mp_atom(x)

        values,_=torch.sort(atom_x.view(-1),descending=True)
        #select the threshold dynamically
        bs,n_atoms,_=x.shape
        threshold=values[bs*n_atoms*self.atom_occuranes].item()

        sorted_tensor, indices = torch.sort(atom_x, dim=2,descending=True)
        indices=indices[:,:,:self.num_indices]
        atom_x=torch.gather(atom_x,2,indices)

        valid_atoms=atom_x>threshold
        valid_atoms=valid_atoms.unsqueeze(-1).repeat(1, 1, 1, indices.shape[1])
        
        pos_enc=self.pos_encoder(indices)
        pos_enc=pos_enc*valid_atoms.int()
        pos_enc=pos_enc.view(pos_enc.shape[0],-1,pos_enc.shape[-1])

        #select the threshold dynamically
        # _,ind=torch.sort(atom_x.view(-1),descending=True)
        # pos_enc_=pos_enc[ind,:][:self.seq_len]
        pos_enc=pos_enc.swapaxes(0,1)
        # d=torch.zeros(128,32,64).double().to(torch.device('cuda'))
        # tr=self.transformer.to(torch.device('cuda'))
        # tr(d)
        output=F.softmax(self.transformer(pos_enc)[-1,:,:],dim=1)
        # output=F.softmax(self.lin_classifier(embs),dim=1)

        # x=self.relu(self.embedding(x_mp))
        # x=self.classifier(x)
        # x=torch.mean(x,dim=2)
        # output=F.softmax(x,dim=1)
            
        return output

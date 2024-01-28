import torch.nn as nn
import torch

def eval(model,dataloader):
    MSE_loss_fn = nn.MSELoss()
    imu_loss_sum,forcast_loss_sum,atom_loss_sum=0,0,0
    for i,input in enumerate(dataloader):
        imu,xyz,imu_mask,xyz_mask,imu_len=input
        output=model(imu,imu_mask,imu_len)
        imu_loss=MSE_loss_fn(imu*imu_mask,output['imu_gen']*imu_mask)
        forcast_loss=MSE_loss_fn(output['forcast_real']*output['forcast_mask'],
                                    output['forcast']*output['forcast_mask'])
        imu_atoms=output['imu_atoms']
        atom_loss=MSE_loss_fn(output['atom_gen']*output['atom_mask'],imu_atoms*output['atom_mask'])
            
        imu_loss_sum+=imu_loss.item()
        forcast_loss_sum+=forcast_loss.item()
        atom_loss_sum+=atom_loss.item()

        # forcast_loss_seg=(torch.mean(torch.square(output['forcast_real']*output['forcast_mask']-output['forcast']*output['forcast_mask']),dim=-1))

        # bs,seq,_,l=imu.shape
        # imu_values=[]
        # forc_loss=[]
        # for b in range(bs):
        #     values=torch.cat([imu[b,j,:,:] for j in range(seq)],dim=-1)
        #     imu_values.append(values)
        #     batch_forcast_loss=forcast_loss_seg[b*seq:(b+1)*seq]
        #     batch_forcast_loss=torch.unsqueeze(batch_forcast_loss,dim=0).unsqueeze(dim=0)
        #     batch_forcast_loss=nn.functional.interpolate(batch_forcast_loss,seq*l)
        #     forc_loss.append(batch_forcast_loss[0][0])  
    output={
        'imu_loss': imu_loss_sum/len(dataloader),
        'forcast_loss':forcast_loss_sum/len(dataloader),
        'atom_loss':atom_loss_sum/len(dataloader)
    }  
    return output

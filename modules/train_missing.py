#%%
import numpy as np
from tqdm import tqdm
import wandb

import torch
from torch import nn
import torch.nn.functional as F
#%%
def train_function(
    model,
    config,
    optimizer, 
    train_dataloader,
    device):
    
    for epoch in range(config["epochs"]):
        logs = {
            'loss': [], 
        }
        
        for batch in tqdm(train_dataloader, desc="inner loop"):
            batch = batch.to(device)
            mask1 = torch.rand(batch.size(0), model.EncodedInfo.num_features) > torch.rand(len(batch), 1)
            mask1 = mask1.to(device)
            nan_mask = batch.isnan()
            mask = mask1 | nan_mask
            loss_mask = mask1 & ~nan_mask
            
            masked_batch = batch.clone()
            masked_batch[mask] = 0. # [MASKED] token
            
            loss_ = []
            
            optimizer.zero_grad()
            
            pred = model(masked_batch)
            
            loss = 0.
            for j in range(model.EncodedInfo.num_features):
                tmp = F.cross_entropy(
                    pred[j][:, 1:][loss_mask[:, j]], # ignore [MASKED] token probability
                    batch[:, j][loss_mask[:, j]].long()-1 # ignore unmasked
                )
                if not tmp.isnan():
                    loss += tmp
                    
            loss.backward()
            optimizer.step()
            
            loss_.append(('loss', loss))
            
            """accumulate losses"""
            for x, y in loss_:
                logs[x] = logs.get(x) + [y.item()]
            
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})

    return
#%%
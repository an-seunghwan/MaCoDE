#%%
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader

from modules.utility import set_random_seed

import numpy as np
import pandas as pd
from tqdm import tqdm
#%% 
class Embedding(nn.Module):
    def __init__(self, config, EncodedInfo, device):
        super().__init__()
        self.config = config
        self.EncodedInfo = EncodedInfo
        self.device = device

        self.ContEmbed = nn.ModuleList()
        for _ in range(EncodedInfo.num_continuous_features):
            self.ContEmbed.append(
                nn.Embedding(config["bins"] + 1, config["dim_transformer"]).to(device)
            )

        self.DiscEmbed = nn.ModuleList()
        for num_category in EncodedInfo.num_categories:
            self.DiscEmbed.append(
                nn.Embedding(num_category + 1, config["dim_transformer"]).to(device)
            )
        
        self.init_weights()
        
    def init_weights(self):
        for layer in self.ContEmbed:
            nn.init.normal_(layer.weight, mean=0.0, std=0.05) 
        for layer in self.DiscEmbed:
            nn.init.normal_(layer.weight, mean=0.0, std=0.05) 

    def forward(self, batch):
        # continuous
        continuous = batch[:, :self.EncodedInfo.num_continuous_features].long()
        continuous_embedded = torch.stack(
            [self.ContEmbed[i](continuous[:, i]) for i in range(continuous.size(1))]
        ).transpose(0, 1) # [batch, num_continuous, dim_transformer]
        
        # discrete
        categorical = batch[:, self.EncodedInfo.num_continuous_features:].long()
        categorical_embedded = torch.stack(
            [self.DiscEmbed[i](categorical[:, i]) for i in range(categorical.size(1))]
        ).transpose(0, 1) # [batch, num_categories, dim_transformer]

        # [batch, num_continuous_features + len(num_categories), dim_transformer]
        embedded = torch.cat([continuous_embedded, categorical_embedded], dim=1)
        return embedded
#%%
class DynamicLinear(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.E = embedding
        self.bias = nn.Parameter(torch.zeros(len(embedding.weight)))

    def forward (self, x):
        h = x @ self.E.weight.T + self.bias
        return h
#%%
class DynamicLinearLayer(nn.Module):
    def __init__(self, config, EncodedInfo, device):
        super().__init__()
        
        self.embedding = nn.ModuleList()
        for _ in range(EncodedInfo.num_continuous_features):
            self.embedding.append(
                nn.Embedding(
                    config["bins"] + 1, 
                    config["dim_transformer"]
                ).to(device)
            )
        for num_category in EncodedInfo.num_categories:
            self.embedding.append(
                nn.Embedding(
                    num_category + 1, 
                    config["dim_transformer"]
                ).to(device)
            )

        self.dynamic_linear = nn.ModuleList()
        for embedding in self.embedding:
            self.dynamic_linear.append(DynamicLinear(embedding).to(device))

    def forward(self, x):
        return [
            self.dynamic_linear[i](x[:, i, :]) for i in range(len(self.dynamic_linear))
        ] 
#%%
class MaCoDE(nn.Module):
    def __init__(self, config, EncodedInfo, device):
        super().__init__()
        self.config = config
        self.EncodedInfo = EncodedInfo
        self.device = device
        
        self.embedding = Embedding(config, EncodedInfo, device).to(device)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.config["dim_transformer"], 
            nhead=self.config['num_transformer_heads'], 
            dropout=self.config["transformer_dropout"], 
            batch_first=True).to(device)
        self.transformer = nn.TransformerEncoder(
            transformer_layer, self.config["num_transformer_layer"])
        
        self.dynamic_linear = DynamicLinearLayer(
            config, EncodedInfo, device)

    def forward(self, batch):
        x = self.embedding(batch)
        x = self.transformer(x)
        pred = self.dynamic_linear(x)
        return pred
    
    def generate_synthetic_data(self, n, train_dataset, tau=1):
        data = []
        batch_size = 64
        steps = n // batch_size + 1
        
        for _ in tqdm(range(steps), desc="Generate Synthetic Dataset..."):
            with torch.no_grad():
                batch = torch.zeros(
                    batch_size, train_dataset.EncodedInfo.num_features
                ).to(self.device)
                mask = torch.ones(
                    batch_size, train_dataset.EncodedInfo.num_features
                ).bool().to(self.device)
                # permute the generation order of columns
                for i in torch.randperm(train_dataset.EncodedInfo.num_features):
                    masked_batch = batch.clone()
                    masked_batch[mask] = 0. # [MASKED] token
                    pred = self(masked_batch)
                    
                    batch[:, i] = Categorical(logits=pred[i][:, 1:] / tau).sample().float() + 1
                    mask[: , i] = False
            data.append(batch)
        
        data = torch.cat(data, dim=0)
        data = data[:n, :]
        
        cont = data.int().cpu().numpy()[:, :train_dataset.EncodedInfo.num_continuous_features]
        quantiles = np.random.uniform(
            low=train_dataset.bins[cont-1],
            high=train_dataset.bins[cont],
        )
        cont = pd.DataFrame(quantiles, columns=train_dataset.continuous_features)
        
        data = pd.DataFrame(data.cpu().numpy(), columns=train_dataset.features)
        for col, cdf in train_dataset.EmpiricalCDFs.items():
            data[col] = cdf.quantile(cont[col])
        data[train_dataset.categorical_features] = data[train_dataset.categorical_features].astype(int)
        data[train_dataset.integer_features] = data[train_dataset.integer_features].round(0).astype(int)
        return data
    
    def impute(self, train_dataset, tau=1, seed=0):
        torch.random.manual_seed(seed)
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False)
        
        imputed = []
        for batch in train_dataloader:
            batch = batch.to(self.device)
            mask = batch.isnan()
            
            with torch.no_grad():
                pred = self(batch.nan_to_num(0))
                
            for i in torch.randperm(train_dataset.EncodedInfo.num_features):
                x = Categorical(logits=pred[i][:, 1:] / tau).sample().float() + 1
                batch[:, i][mask[:, i]] = x[mask[:, i]]
            imputed.append(batch)
            
        data = torch.cat(imputed, dim=0)
        
        cont = data.int().cpu().numpy()[:, :train_dataset.EncodedInfo.num_continuous_features]
        quantiles = np.random.uniform(
            low=train_dataset.bins[cont-1],
            high=train_dataset.bins[cont],
        )
        cont = pd.DataFrame(quantiles, columns=train_dataset.continuous_features)
        
        data = pd.DataFrame(data.cpu().numpy(), columns=train_dataset.features)
        for col, cdf in train_dataset.EmpiricalCDFs.items():
            data[col] = cdf.quantile(cont[col])
        data[train_dataset.categorical_features] = data[train_dataset.categorical_features].astype(int)
        data[train_dataset.integer_features] = data[train_dataset.integer_features].round(0).astype(int)
        
        # Impute missing values
        syndata = data * train_dataset.mask + train_dataset.raw_data * (1. - train_dataset.mask)
        return syndata
#%%
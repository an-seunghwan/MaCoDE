# %%
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from macode.dataset import CustomDataset
from macode.model import Model


# %%
class MaCoDE(nn.Module):
    def __init__(
        self,
        data: pd.DataFrame,
        continuous_features=[],
        categorical_features=[],
        integer_features=[],
        seed: int = 0,
        bins: int = 100,
        dim_transformer: int = 128,
        num_transformer_heads: int = 8,
        num_transformer_layer: int = 2,
        epochs: int = 1000,
        batch_size: int = 1024,
        lr: float = 0.001,
        device="cpu",
    ):
        """
        Args:
            data (pd.DataFrame): the observed tabular dataset
            continuous_features (list, optional): the list of continuous columns of data. Defaults to [].
                - If it is [], then all columns of data will be treated as continuous column
            categorical_features (list, optional): the list of categorical columns of data. Defaults to [].
                - If it is [], all other columns except continuous columns will be categorical column.
            integer_features (list, optional): the list of integer-type columns of data. Defaults to [].

            seed (int, optional): seed for repeatable results. Defaults to 0.
            bins (int, optional): the number of bins for discretization. Defaults to 100.
            dim_transformer (int, optional): the embedding size (input dimension size of transformer). Defaults to 128.
            num_transformer_heads (int, optional): the number of heads in transformer. Defaults to 8.
            num_transformer_layer (int, optional): the number of layers in transformer. Defaults to 2.
            
            epochs (int, optional): the number of epochs. Defaults to 1000.
            batch_size (int, optional): the batch size. Defaults to 1024.
            lr (float, optional): learning rate. Defaults to 0.001.
            device (str, optional): Defaults to "cpu".
        """

        super(MaCoDE, self).__init__()

        self.seed = seed
        self.bins = bins
        self.dim_transformer = dim_transformer
        self.num_transformer_heads = num_transformer_heads
        self.num_transformer_layer = num_transformer_layer
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

        self.dataset = CustomDataset(
            data=data,
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            integer_features=integer_features,
        )
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )
        self.EncodedInfo = self.dataset.EncodedInfo

        self.set_random_seed(self.seed)
        self.initialize()

    def set_random_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        return

    def initialize(self):
        self.model = Model(
            EncodedInfo=self.EncodedInfo,  # information of the dataset
            bins=self.bins,  # the number of bins for discretization
            dim_transformer=self.dim_transformer,  # the embedding size (input dimension size of transformer)
            num_transformer_heads=self.num_transformer_heads,  # the number of heads in transformer
            num_transformer_layer=self.num_transformer_layer,  # the number of layers in transformer
            device="cpu",
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr,
            weight_decay=0.)
        return
    
    def multiclass_loss(self, batch, pred, mask):
        class_loss = 0.
        for j in range(len(pred)):
            tmp = F.cross_entropy(
                pred[j][mask[:, j]], # ignore [MASKED] token probability
                batch[:, j][mask[:, j]].long()-1 # ignore unmasked
            )
            if not tmp.isnan():
                class_loss += tmp
        return class_loss

    def train(self):
        self.set_random_seed(self.seed)

        for epoch in tqdm(range(self.epochs), desc="Training..."):
            logs = {}
        
            for batch in self.dataloader:
                batch = batch.to(self.device)
                
                mask1 = torch.rand(batch.size(0), self.EncodedInfo.num_features) > torch.rand(len(batch), 1)
                mask1 = mask1.to(self.device)
                nan_mask = batch.isnan()
                mask = mask1 | nan_mask
                loss_mask = mask1 & ~nan_mask
                
                masked_batch = batch.clone()
                masked_batch[mask] = 0. # [MASKED] token
                
                loss_ = []
                
                self.optimizer.zero_grad()
                
                pred = self.model(masked_batch)
                
                loss = self.multiclass_loss(batch, pred, loss_mask)
                loss_.append(('loss', loss))
                
                loss.backward()
                self.optimizer.step()
                
                for x, y in loss_:
                    try:
                        logs[x] = logs.get(x) + [y.item()]
                    except:
                        logs[x] = []
                        logs[x] = logs.get(x) + [y.item()]

            print_input = "[epoch {:03d}]".format(epoch + 1)
            print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
            print(print_input)

        return

    def generate_data(
        self, 
        n: int, 
        tau: float = 1,
        seed: int = 0,
    ):
        """
        Args:
            n (int): the number of synthetic samples to generate
            tau (float, optional): the hyper-parameter for privacy control (it must be positive). Defaults to 1.
            seed (int, optional): seed for repeatable results. Defaults to 0.
        """
        if tau <= 0:
            ValueError(r"$\tau$ must be positive!")
        
        self.set_random_seed(seed)
        data = []
        batch_size = 64
        steps = n // batch_size + 1
        
        for _ in tqdm(range(steps), desc="Generate Synthetic Dataset..."):
            with torch.no_grad():
                batch = torch.zeros(
                    batch_size, self.EncodedInfo.num_features
                ).to(self.device)
                mask = torch.ones(
                    batch_size, self.EncodedInfo.num_features
                ).bool().to(self.device)
                # permute the generation order of columns
                for i in torch.randperm(self.EncodedInfo.num_features):
                    masked_batch = batch.clone()
                    masked_batch[mask] = 0. # [MASKED] token
                    pred = self.model(masked_batch)
                    batch[:, i] = Categorical(logits=pred[i] / tau).sample().float() + 1
                    mask[: , i] = False
            data.append(batch)
        
        data = torch.cat(data, dim=0)
        data = data[:n, :]
        
        cont = data.int().cpu().numpy()[:, :self.EncodedInfo.num_continuous_features]
        quantiles = np.random.uniform(
            low=self.dataset.bins[cont-1],
            high=self.dataset.bins[cont],
        )
        cont = pd.DataFrame(quantiles, columns=self.dataset.continuous_features)
        
        data = pd.DataFrame(data.cpu().numpy(), columns=self.dataset.features)
        for col, scaler in self.dataset.scalers.items():
            data[[col]] = scaler.inverse_transform(cont[[col]]).astype(np.float64)
            
        data[self.dataset.categorical_features] = data[self.dataset.categorical_features].astype(int)
        data[self.dataset.integer_features] = data[self.dataset.integer_features].round(0).astype(int)
        return data
    
    def impute(
        self, 
        tau: float = 1
    ):
        """
        Args:
            tau (float, optional): the hyper-parameter for privacy control (it must be positive). Defaults to 1.
        """
        train_dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=False)
        
        imputed = []
        for batch in train_dataloader:
            batch = batch.to(self.device)
            mask = batch.isnan()
            
            with torch.no_grad():
                pred = self.model(batch.nan_to_num(0))
                
            for i in range(self.EncodedInfo.num_features):
                x = Categorical(logits=pred[i] / tau).sample().float() + 1
                batch[:, i][mask[:, i]] = x[mask[:, i]]
            imputed.append(batch)
            
        data = torch.cat(imputed, dim=0)
        
        cont = data.int().cpu().numpy()[:, :self.EncodedInfo.num_continuous_features]
        quantiles = np.random.uniform(
            low=self.dataset.bins[cont-1],
            high=self.dataset.bins[cont],
        )
        cont = pd.DataFrame(quantiles, columns=self.dataset.continuous_features)
        
        data = pd.DataFrame(data.cpu().numpy(), columns=self.dataset.features)
        for col, scaler in self.dataset.scalers.items():
            data[[col]] = scaler.inverse_transform(cont[[col]]).astype(np.float64)
            
        data[self.dataset.categorical_features] = data[self.dataset.categorical_features].astype(int)
        data[self.dataset.integer_features] = data[self.dataset.integer_features].round(0).astype(int)
        
        # Impute missing values
        mask_ = self.dataset.raw_data.isna().values
        imputed = pd.DataFrame(
            data.values * mask_ + np.nan_to_num(self.dataset.raw_data.values, 0.) * (1. - mask_),
            columns=self.dataset.raw_data.columns)
        return imputed
# %%

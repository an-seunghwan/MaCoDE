#%%
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import namedtuple
from tqdm import tqdm

from modules.missing import generate_mask
from datasets.raw_data import load_raw_data

EncodedInfo = namedtuple(
    'EncodedInfo', 
    ['num_features', 'num_continuous_features', 'num_categories'])
#%%
"""
Data Source: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
"""

class EmpiricalCDF:
    def __init__(self, data):
        self.data = np.sort(data)
        self.n = len(data)
        self.cdf_values = np.arange(1, self.n + 1) / self.n

    def cdf(self, x):
        # Count the number of data points less than or equal to x
        cdf_values = np.searchsorted(self.data, x, side='right') / self.n
        return cdf_values

    def quantile(self, q):
        # Use np.interp for interpolation between CDF values and data points
        inverse_cdf_values = np.interp(q, self.cdf_values, self.data)
        return inverse_cdf_values
#%%
class CustomDataset(Dataset):
    def __init__(self, config):
        
        self.config = config
        data, continuous_features, categorical_features, integer_features, ClfTarget = load_raw_data(config["dataset"])
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.integer_features = integer_features
        self.ClfTarget = ClfTarget
        
        self.features = self.continuous_features + self.categorical_features
        self.col_2_idx = {col : i + 1 for i, col in enumerate(data[self.features].columns.to_list())}
        self.num_continuous_features = len(self.continuous_features)
        
        # encoding categorical dataset
        data[self.categorical_features] = data[self.categorical_features].apply(
            lambda col: col.astype('category').cat.codes + 1)
        self.num_categories = data[self.categorical_features].nunique(axis=0).to_list()

        data = data[self.features] # select features for training
        data = data.reset_index(drop=True)
        self.raw_data = data[self.features]
        
        # Apply missing data mechanism
        if config["missing_type"] != "None":
            mask = generate_mask(
                torch.from_numpy(data.values).float(), 
                config["missing_rate"], 
                config["missing_type"],
                seed=config["seed"])
            data.mask(mask.astype(bool), np.nan, inplace=True)
            self.mask = mask
        
        self.EmpiricalCDFs = {}
        self.bins = np.linspace(0, 1, self.config["bins"]+1, endpoint=True)
        print(f"The number of bins: {len(self.bins)-1}")
        transformed = []
        for continuous_feature in tqdm(self.continuous_features, desc="Tranform Continuous Features..."):
            transformed.append(self.transform_continuous(data, continuous_feature))
        
        self.data = np.concatenate(
            transformed + [data[self.categorical_features].values], axis=1
        )
        
        self.EncodedInfo = EncodedInfo(
            len(self.features), self.num_continuous_features, self.num_categories)
        
    def transform_continuous(self, data, col):
        nan_value = data[[col]].to_numpy().astype(float)
        nan_mask = np.isnan(nan_value)
        feature = nan_value[~nan_mask].reshape(-1, 1)
        
        density = EmpiricalCDF(feature[:, 0])
        self.EmpiricalCDFs[col] = density
            
        nan_value[nan_mask] = 0.
        nan_value = np.digitize(
            np.where(
                density.cdf(nan_value) == 1,
                1 - 1e-6,
                density.cdf(nan_value)
            ),
            self.bins
        ).astype(float)
        
        nan_value[nan_mask] = np.nan # reverse originally NaN values
        return nan_value

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])
#%%

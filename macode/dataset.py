# %%
from collections import namedtuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import Dataset
from tqdm import tqdm

EncodedInfo = namedtuple(
    "EncodedInfo", ["num_features", "num_continuous_features", "num_categories"]
)


#%%
class CustomDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        continuous_features=[],
        categorical_features=[],
        integer_features=[],
        bins=100,
    ):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if len(set(continuous_features) - set(data.columns)) > 0:
            raise ValueError("There exist invalid continuous column names.")

        if len(set(categorical_features) - set(data.columns)) > 0:
            raise ValueError("There exist invalid categorical column names.")

        if len(set(integer_features) - set(data.columns)) > 0:
            raise ValueError("There exist invalid integer column names.")

        if (len(continuous_features) == 0) and (len(categorical_features) == 0):
            continuous_features = list(data.columns)
        elif len(continuous_features) == 0:
            continuous_features = [
                x for x in data.columns if x not in categorical_features
            ]
        else:
            categorical_features = [
                x for x in data.columns if x not in continuous_features
            ]

        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.integer_features = integer_features

        self.features = self.continuous_features + self.categorical_features
        self.col_2_idx = {
            col: i for i, col in enumerate(data[self.features].columns.to_list())
        }
        self.num_continuous_features = len(self.continuous_features)

        # encoding categorical dataset
        data[self.categorical_features] = data[self.categorical_features].apply(
            lambda col: col.astype("category").cat.codes.where(col.notna(), np.nan) + 1 # "0" for [MASK] token
        )
        self.num_categories = data[self.categorical_features].nunique(axis=0).to_list()

        data = data[self.features]
        data = data.reset_index(drop=True)
        self.raw_data = data[self.features]

        self.scalers = {}
        self.bins = np.linspace(0, 1, bins+1, endpoint=True)
        print(f"The number of bins: {len(self.bins)-1}")
        transformed = []
        for continuous_feature in tqdm(
            self.continuous_features, desc="Tranform Continuous Features..."
        ):
            transformed.append(self.transform_continuous(data, continuous_feature))

        self.data = np.concatenate(
            transformed + [data[self.categorical_features].values], axis=1
        )

        self.EncodedInfo = EncodedInfo(
            len(self.features), self.num_continuous_features, self.num_categories
        )

    def transform_continuous(self, data, col):
        nan_value = data[[col]].to_numpy().astype(float)
        nan_mask = np.isnan(nan_value)
        feature = nan_value[~nan_mask].reshape(-1, 1)
        
        scaler = QuantileTransformer(
            n_quantiles=10000,
            output_distribution='uniform',
            subsample=None,
        ).fit(feature)
        self.scalers[col] = scaler
        
        nan_value[nan_mask] = 0. # replace NaN with arbitrary value
        transformed = scaler.transform(nan_value)
        transformed = np.where(
            transformed == 1, 1-1e-6, transformed
        ) # maximum value will be assinged to the last bin
        transformed = np.digitize(
            transformed, self.bins
        ).astype(float) # range = (1, 2, ..., #bins) ("0" for [MASK] token) 
        transformed[nan_mask] = np.nan
        return transformed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])


# %%

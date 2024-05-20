#%%
import torch

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
#%%
"""for reproducibility"""
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)   
#%%
def get_frequency(
    X_gt: pd.DataFrame, X_synth: pd.DataFrame, n_histogram_bins: int = 10
):
    """
    Reference:
    [1] https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/metrics/_utils.py
    
    Get percentual frequencies for each possible real categorical value.

    Returns:
        The observed and expected frequencies (as a percent).
    """
    res = {}
    for col in X_gt.columns:
        local_bins = min(n_histogram_bins, len(X_gt[col].unique()))

        if len(X_gt[col].unique()) < 5:  # categorical
            gt = (X_gt[col].value_counts() / len(X_gt)).to_dict()
            synth = (X_synth[col].value_counts() / len(X_synth)).to_dict()
        else:
            gt_vals, bins = np.histogram(X_gt[col], bins=local_bins)
            synth_vals, _ = np.histogram(X_synth[col], bins=bins)
            gt = {k: v / (sum(gt_vals) + 1e-8) for k, v in zip(bins, gt_vals)}
            synth = {k: v / (sum(synth_vals) + 1e-8) for k, v in zip(bins, synth_vals)}

        for val in gt:
            if val not in synth or synth[val] == 0:
                synth[val] = 1e-11
        for val in synth:
            if val not in gt or gt[val] == 0:
                gt[val] = 1e-11

        if gt.keys() != synth.keys():
            raise ValueError(f"Invalid features. {gt.keys()}. syn = {synth.keys()}")
        res[col] = (list(gt.values()), list(synth.values()))

    return res
#%%
def marginal_plot(train, syndata, config):
    if not os.path.exists(f"./assets/figs/{config['dataset']}/seed{config['seed']}/"):
        os.makedirs(f"./assets/figs/{config['dataset']}/seed{config['seed']}/")
    
    figs = []
    for idx, feature in tqdm(enumerate(train.columns), desc="Plotting Histograms..."):
        fig = plt.figure(figsize=(7, 4))
        fig, ax = plt.subplots(1, 1)
        sns.histplot(
            data=syndata,
            x=syndata[feature],
            stat='density',
            label='synthetic',
            ax=ax,
            bins=int(np.sqrt(len(syndata)))) 
        sns.histplot(
            data=train,
            x=train[feature],
            stat='density',
            label='train',
            ax=ax,
            bins=int(np.sqrt(len(train)))) 
        ax.legend()
        ax.set_title(f'{feature}', fontsize=15)
        plt.tight_layout()
        plt.savefig(f"./assets/figs/{config['dataset']}/seed{config['seed']}/hist_{feature}.png")
        # plt.show()
        plt.close()
        figs.append(fig)
    return figs
#%%
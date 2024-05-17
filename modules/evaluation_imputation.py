# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import namedtuple

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")

Metrics = namedtuple(
    "Metrics",
    [
        "bias", "coverage", "interval"
    ],
)
#%%
def evaluate(train_dataset, model, M=100, tau=1):
    """target estimand"""
    data = train_dataset.raw_data[train_dataset.continuous_features]
    true = (data > data.mean(axis=0)).astype(float).mean(axis=0)
    # true = pd.DataFrame(train_dataset.raw_data[train_dataset.continuous_features].mean(axis=0))
    
    """multiple imputation"""
    est = []
    var = []
    for s in tqdm(range(M), desc="Multiple Imputation..."):
        imputed = model.impute(train_dataset, tau=tau, seed=s)
        
        data = imputed[train_dataset.continuous_features]
        binary = (data > data.mean(axis=0)).astype(float)
        p = binary.mean(axis=0)
        est.append(p)
        var.append(p * (1. - p) / len(binary))
        # est.append(
        #     pd.DataFrame(imputed[train_dataset.continuous_features].mean(axis=0))
        # )
        # var.append(
        #     pd.DataFrame(imputed[train_dataset.continuous_features].var(axis=0, ddof=1) / len(imputed))
        # )
        
    Q = np.mean(est, axis=0)
    U = np.mean(var, axis=0) + (M + 1) / M * np.var(est, axis=0, ddof=1)
    lower = Q - 1.96 * np.sqrt(U)
    upper = Q + 1.96 * np.sqrt(U)
    
    bias = float(np.abs(Q - true).mean())
    coverage = float(((lower < true) & (true < upper)).mean())
    interval = float((upper - lower).mean())
    
    return Metrics(
        bias, coverage, interval
    )
#%%
# def evaluate(train_dataset, model, M=10, tau=1):
#     target = train_dataset.ClfTarget
    
#     """target estimand"""
#     data = train_dataset.raw_data.copy()
#     mean = data[train_dataset.continuous_features].mean()
#     std = data[train_dataset.continuous_features].std()
#     data[train_dataset.continuous_features] -= mean
#     data[train_dataset.continuous_features] /= std
#     data = pd.get_dummies(
#         data, 
#         columns=[x for x in train_dataset.categorical_features if x != target], 
#         drop_first=True,
#         dtype=float)
#     covariates = [x for x in data.columns if x != target]
#     # clf = LogisticRegression(random_state=0, max_iter=5000, n_jobs=-1)
#     clf = svm.SVC(kernel='rbf', random_state=0)
#     scores = cross_val_score(clf, data[covariates], data[target], cv=5)
#     true = np.mean(scores)
    
#     """multiple imputation"""
#     est = []
#     var = []
#     for s in tqdm(range(M), desc="Multiple Imputation..."):
#         imputed = model.impute(train_dataset, tau=tau, seed=s)
        
#         mean = imputed[train_dataset.continuous_features].mean()
#         std = imputed[train_dataset.continuous_features].std()
#         imputed[train_dataset.continuous_features] -= mean
#         imputed[train_dataset.continuous_features] /= std
#         imputed = pd.get_dummies(
#             imputed, 
#             columns=[x for x in train_dataset.categorical_features if x != target], 
#             drop_first=True,
#             dtype=float)
#         covariates = [x for x in imputed.columns if x != target]
#         # clf = LogisticRegression(random_state=0, max_iter=5000, n_jobs=-1)
#         clf = svm.SVC(kernel='rbf', random_state=0)
#         scores = cross_val_score(clf, imputed[covariates], imputed[target], cv=5)
#         est.append(np.mean(scores))
#         var.append(np.var(scores) / 5)
        
#     Q = np.mean(est, axis=0)
#     U = np.mean(var, axis=0) + (M + 1) / M * np.var(est, axis=0, ddof=1)
#     lower = Q - 1.96 * np.sqrt(U)
#     upper = Q + 1.96 * np.sqrt(U)
    
#     bias = float(np.abs(Q - true).mean())
#     coverage = float(((lower < true) & (true < upper)).mean())
#     interval = float((upper - lower).mean())
    
#     return Metrics(
#         bias, coverage, interval
#     )
#%%
# def evaluate(train_dataset, target, model, M=10, tau=1):
#     """target estimand"""
#     true = train_dataset.raw_data[target].mean()
    
#     """multiple imputation"""
#     est = []
#     var = []
#     for s in tqdm(range(M), desc="Multiple Imputation..."):
#         imputed = model.impute(train_dataset, tau=tau, seed=s)
#         est.append(imputed[target].mean())
#         var.append(imputed[target].var(ddof=1) / len(imputed))
    
#     Q = np.mean(est)
#     U = np.mean(var) + (M + 1) / M * np.var(est, ddof=1)
#     lower = Q - 1.96 * np.sqrt(U)
#     upper = Q + 1.96 * np.sqrt(U)
    
#     bias = np.abs(Q - true)
#     coverage = float((lower < true) & (true < upper))
#     interval = upper - lower
    
#     return Metrics(
#         bias, coverage, interval
#     )
#%%
# def evaluate(train_dataset, target, model, M=10, tau=1):
#     """with complete data"""
#     data = pd.get_dummies(
#         train_dataset.raw_data, 
#         columns=[x for x in train_dataset.categorical_features if x != target],
#         drop_first=True,
#         dtype=float)
#     covariates = [x for x in data.columns if x != target]
#     scaling = [x for x in train_dataset.continuous_features if x != target]
    
#     data[target] = pd.cut(data[target], 2, labels=[0, 1])
#     data[scaling] -= data[scaling].mean()
#     data[scaling] /= data[scaling].std()
    
#     # logit = sm.Logit(data[target], sm.add_constant(data[covariates])).fit(disp=0)
#     # true = pd.DataFrame(logit.params).drop(index="const")
    
#     logit = LogisticRegression(random_state=42)
#     scores = cross_val_score(logit, data[covariates], data[target], cv=5)
#     true = np.mean(scores)
    
#     """with imputed data"""
#     est = []
#     lower = []
#     upper = []
#     for s in tqdm(range(M), desc="Multiple Imputation..."):
#         imputed = model.impute(train_dataset, tau=tau, seed=s)
        
#         imputed = pd.get_dummies(
#             imputed, 
#             columns=[x for x in train_dataset.categorical_features if x != target],
#             drop_first=True,
#             dtype=float)
        
#         imputed[target] = pd.cut(imputed[target], 2, labels=[0, 1])
#         imputed[scaling] -= imputed[scaling].mean()
#         imputed[scaling] /= imputed[scaling].std()
        
#         # logit = sm.Logit(imputed[target], sm.add_constant(imputed[covariates])).fit(disp=0)
#         # est.append(pd.DataFrame(logit.params).drop(index="const"))
#         # lower.append(pd.DataFrame(logit.conf_int(alpha=0.05)[0]).drop(index="const"))
#         # upper.append(pd.DataFrame(logit.conf_int(alpha=0.05)[1]).drop(index="const"))
        
#         logit = LogisticRegression(random_state=42)
#         scores = cross_val_score(logit, imputed[covariates], imputed[target], cv=5)
#         mean = np.mean(scores)
#         std = np.std(scores)
#         l = mean - 1.96 * std
#         u = mean + 1.96 * std
#         est.append(mean)
#         lower.append(l)
#         upper.append(u)
    
#     # bias = np.abs(np.mean(est) - true).mean().item()
#     # coverage = ((np.mean(lower, axis=0) < true.values) & (true.values < np.mean(upper, axis=0))).mean()
#     # interval = (np.mean(upper, axis=0) - np.mean(lower, axis=0)).mean()
    
#     bias = np.abs(np.mean(est) - true)
#     coverage = float((np.mean(lower) < true) and (true < np.mean(upper)))
#     interval = np.mean(upper) - np.mean(lower)
    
#     return Metrics(
#         bias, coverage, interval
#     )
#%%
# def evaluate(train_dataset, target, model, M=10, tau=1):
#     scaling = [x for x in train_dataset.continuous_features if x != target]
    
#     """with complete data"""
#     data = train_dataset.raw_data
#     data[scaling] -= data[scaling].mean()
#     data[scaling] /= data[scaling].std()
    
#     covariates = [x for x in data.columns if x != target]
#     reg = sm.OLS(data[target], sm.add_constant(data[covariates])).fit()
    
#     true = pd.DataFrame(reg.params).drop(index="const")
    
#     """with imputed data"""
#     est = []
#     lower = []
#     upper = []
#     for s in tqdm(range(M), desc="Multiple Imputation..."):
#         imputed = model.impute(train_dataset, tau=tau, seed=s)
#         imputed[scaling] -= imputed[scaling].mean()
#         imputed[scaling] /= imputed[scaling].std()
        
#         covariates = [x for x in data.columns if x != target]
#         reg = sm.OLS(data[target], sm.add_constant(data[covariates])).fit()
        
#         est.append(pd.DataFrame(reg.params).drop(index="const"))
#         lower.append(pd.DataFrame(reg.conf_int(alpha=0.05)[0]).drop(index="const"))
#         upper.append(pd.DataFrame(reg.conf_int(alpha=0.05)[1]).drop(index="const"))
    
#     bias = np.abs(np.mean(est) - true).mean().item()
#     coverage = ((np.mean(lower, axis=0) < true.values) & (true.values < np.mean(upper, axis=0))).mean()
#     interval = (np.mean(upper, axis=0) - np.mean(lower, axis=0)).mean()
    
#     return Metrics(
#         bias, coverage, interval
#     )
#%%
#%%
"""
Reference:
[1] Reimagining Synthetic Tabular Data Generation through Data-Centric AI: A Comprehensive Benchmark
- https://github.com/HLasse/data-centric-synthetic-data
"""
#%%
import numpy as np

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.stats import spearmanr
#%%
def MLu_reg(train_dataset, test_dataset, syndata):
    train = train_dataset.raw_data.copy()
    test = test_dataset.raw_data.copy()
    syndata = syndata.copy()
    
    """Baseline"""
    print(f"\n(Baseline) Regression: SMAPE...")
    result = []
    for col in train_dataset.continuous_features:
        covariates = [x for x in train.columns if x not in [col]]
        
        regr = RandomForestRegressor(random_state=0, n_jobs=-1)
        regr.fit(train[covariates], train[col])
        pred = regr.predict(test[covariates])
        true = np.array(test[col])
        
        smape = np.abs(true - pred)
        smape /= (np.abs(true) + np.abs(pred)) + 1e-6 # numerical stability
        smape = smape.mean()
        
        result.append((col, smape))
        print("[{}] SMAPE: {:.3f}".format(col, smape))
    base_reg = np.mean([x[1] for x in result])
    
    """Synthetic"""
    print(f"\n(Synthetic) Regression: SMAPE...")
    result = []
    for col in train_dataset.continuous_features:
        covariates = [x for x in syndata.columns if x not in [col]]
        
        regr = RandomForestRegressor(random_state=0, n_jobs=-1)
        regr.fit(syndata[covariates], syndata[col])
        pred = regr.predict(test[covariates])
        true = np.array(test[col])
        
        smape = np.abs(true - pred)
        smape /= (np.abs(true) + np.abs(pred)) + 1e-6 # numerical stability
        smape = smape.mean()
        
        result.append((col, smape))
        print("[{}] SMAPE: {:.3f}".format(col, smape))
    syn_reg = np.mean([x[1] for x in result])
    
    return base_reg, syn_reg
#%%
def MLu_reg_withmissing(test_dataset, syndata):
    test = test_dataset.raw_data.copy()
    syndata = syndata.copy()
    
    """Synthetic"""
    print(f"\n(Synthetic) Regression: SMAPE...")
    result = []
    for col in test_dataset.continuous_features:
        covariates = [x for x in syndata.columns if x not in [col]]
        
        regr = RandomForestRegressor(random_state=0, n_jobs=-1)
        regr.fit(syndata[covariates], syndata[col])
        pred = regr.predict(test[covariates])
        true = np.array(test[col])
        
        smape = np.abs(true - pred)
        smape /= (np.abs(true) + np.abs(pred)) + 1e-6 # numerical stability
        smape = smape.mean()
        
        result.append((col, smape))
        print("[{}] SMAPE: {:.3f}".format(col, smape))
    syn_reg = np.mean([x[1] for x in result])
    
    return syn_reg
#%%
def MLu_cls(train_dataset, test_dataset, syndata):
    continuous = train_dataset.continuous_features
    target = train_dataset.ClfTarget
    
    train_ = train_dataset.raw_data.copy()
    test_ = test_dataset.raw_data.copy()
    syndata_ = syndata.copy()
    
    mean = train_[continuous].mean()
    std = train_[continuous].std()
    train_[continuous] -= mean
    train_[continuous] /= std
    test_[continuous] -= mean
    test_[continuous] /= std
    syndata_[continuous] -= mean
    syndata_[continuous] /= std
    
    covariates = [x for x in train_.columns if x not in [target]]

    """Baseline"""
    performance = []
    print(f"\n(Baseline) Classification: F1...")
    for name, clf in [
        ('logit', LogisticRegression(random_state=0, n_jobs=-1, max_iter=1000)),
        ('GaussNB', GaussianNB()),
        ('KNN', KNeighborsClassifier(n_jobs=-1)),
        ('tree', DecisionTreeClassifier(random_state=0)),
        ('RF', RandomForestClassifier(random_state=0)),
    ]:
        clf.fit(train_[covariates], train_[target])
        pred = clf.predict(test_[covariates])
        f1 = f1_score(test_[target], pred, average='micro')
        if name == "RF":
            feature = [(x, y) for x, y in zip(covariates, clf.feature_importances_)]
        print(f"[{name}] F1: {f1:.3f}")
        performance.append((name, f1))

    base_performance = performance
    base_cls = np.mean([x[1] for x in performance])
    base_feature = feature
    
    """Synthetic"""
    performance = []
    print(f"\n(Synthetic) Classification: F1...")
    for name, clf in [
        ('logit', LogisticRegression(random_state=0, n_jobs=-1, max_iter=1000)),
        ('GaussNB', GaussianNB()),
        ('KNN', KNeighborsClassifier(n_jobs=-1)),
        ('tree', DecisionTreeClassifier(random_state=0)),
        ('RF', RandomForestClassifier(random_state=0)),
    ]:
        clf.fit(syndata_[covariates], syndata_[target])
        pred = clf.predict(test_[covariates])
        f1 = f1_score(test_[target], pred, average='micro')
        if name == "RF":
            feature = [(x, y) for x, y in zip(covariates, clf.feature_importances_)]
        print(f"[{name}] F1: {f1:.3f}")
        performance.append((name, f1))
            
    syn_cls = np.mean([x[1] for x in performance])
    model_selection = spearmanr(
        np.array([x[1] for x in base_performance]),
        np.array([x[1] for x in performance])).statistic
    feature_selection = spearmanr(
        np.array([x[1] for x in base_feature]),
        np.array([x[1] for x in feature])).statistic
    
    return (
        base_cls, syn_cls, model_selection, feature_selection
    )
#%%
def MLu_cls_withmissing(test_dataset, syndata):
    continuous = test_dataset.continuous_features
    target = test_dataset.ClfTarget
    
    test_ = test_dataset.raw_data.copy()
    syndata_ = syndata.copy()
    
    mean = syndata_[continuous].mean()
    std = syndata_[continuous].std()
    test_[continuous] -= mean
    test_[continuous] /= std
    syndata_[continuous] -= mean
    syndata_[continuous] /= std
    
    covariates = [x for x in syndata_.columns if x not in [target]]

    """Synthetic"""
    performance = []
    print(f"(Synthetic) Target: {target}")
    for name, clf in [
        ('logit', LogisticRegression(random_state=0, n_jobs=-1, max_iter=1000)),
        ('GaussNB', GaussianNB()),
        ('KNN', KNeighborsClassifier(n_jobs=-1)),
        ('tree', DecisionTreeClassifier(random_state=0)),
        ('RF', RandomForestClassifier(random_state=0)),
    ]:
        clf.fit(syndata_[covariates], syndata_[target])
        pred = clf.predict(test_[covariates])
        f1 = f1_score(test_[target], pred, average='micro')
        print(f"[{name}] F1: {f1:.3f}")
        performance.append((name, f1))
            
    cls_performance = np.mean([x[1] for x in performance])
    
    return cls_performance
#%%
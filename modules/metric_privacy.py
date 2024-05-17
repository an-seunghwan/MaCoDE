#%%
"""
Reference:
[1] Synthcity: facilitating innovative use cases of synthetic data in different data modalities
- https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/metrics/eval_privacy.py
"""
#%%
import numpy as np
import pandas as pd
from collections import Counter

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix
#%%
def kAnonymization(train_dataset, syndata):
    """
    Returns the k-anonimity ratio between the real data and the synthetic data.
    For each dataset, it is computed the value k which satisfies the k-anonymity rule: 
    each record is similar to at least another k-1 other records on the potentially identifying variables.
    
    For comparison between different datasets, we normalize using the number of observations.
    : higher is better
    """
    
    train = train_dataset.raw_data
    train = train[train_dataset.continuous_features]
    syndata = syndata[train_dataset.continuous_features]
    
    scaler = StandardScaler().fit(train)
    train = scaler.transform(train)
    syndata = scaler.transform(syndata)
    
    def evaluate_data(data):
        values = [999]
        for n_clusters in [2, 5, 10, 15]:
            if len(data) / n_clusters < 10:
                continue
            cluster = KMeans(
                n_clusters=n_clusters, init="k-means++", random_state=0
            ).fit(data)
            counts: dict = Counter(cluster.labels_)
            values.append(np.min(list(counts.values())))
        return np.min(values) / len(data) * 100
    
    return (
        evaluate_data(train),
        evaluate_data(syndata)
    )
#%%
def kMap(train_dataset, syndata):
    """
    Returns the minimum value k that satisfies the k-map rule.
    The data satisfies k-map if every combination of values for the quasi-identifiers appears at least k times in the reidentification(synthetic) dataset.
    
    For comparison between different datasets, we normalize using the number of observations.
    : higher is better
    """
    
    train = train_dataset.raw_data
    train = train[train_dataset.continuous_features]
    syndata = syndata[train_dataset.continuous_features]
    
    scaler = StandardScaler().fit(train)
    train = scaler.transform(train)
    syndata = scaler.transform(syndata)
    
    values = []
    for n_clusters in [2, 5, 10, 15]:
        if len(train) / n_clusters < 10:
            continue
        cluster = KMeans(
            n_clusters=n_clusters, init="k-means++", random_state=0
        ).fit(train)
        clusters = cluster.predict(syndata)
        counts: dict = Counter(clusters)
        values.append(np.min(list(counts.values())))

    if len(values) == 0:
        return 0

    return np.min(values) / len(syndata) * 100
#%%
def DCR_metric(train_dataset, syndata, data_percent=15):
    """
    Reference:
    [1] https://github.com/Team-TUD/CTAB-GAN/blob/main/model/eval/evaluation.py
    
    Returns Distance to Closest Record
    
    Inputs:
    1) train -> real data
    2) synthetic -> corresponding synthetic data
    3) data_percent -> percentage of data to be sampled from real and synthetic datasets for computing Distance to Closest Record
    Outputs:
    1) List containing the 5th percentile distance to closest record (DCR) between real and synthetic as well as within real and synthetic datasets
    along with 5th percentile of nearest neighbour distance ratio (NNDR) between real and synthetic as well as within real and synthetic datasets
    
    : higher is better
    """
    
    train = train_dataset.raw_data
    train = train[train_dataset.continuous_features]
    syndata = syndata[train_dataset.continuous_features]
    
    scaler = StandardScaler().fit(train)
    train = pd.DataFrame(scaler.transform(train), columns=train.columns)
    syndata = pd.DataFrame(scaler.transform(syndata), columns=syndata.columns)
    
    # Sampling smaller sets of real and synthetic data to reduce the time complexity of the evaluation
    real_sampled = train.sample(n=int(len(train)*(.01*data_percent)), random_state=42).to_numpy()
    fake_sampled = syndata.sample(n=int(len(syndata)*(.01*data_percent)), random_state=42).to_numpy()

    # Computing pair-wise distances between real and synthetic 
    dist_rf = metrics.pairwise_distances(real_sampled, Y=fake_sampled, metric='minkowski', n_jobs=-1)
    # Computing pair-wise distances within real 
    dist_rr = metrics.pairwise_distances(real_sampled, Y=None, metric='minkowski', n_jobs=-1)
    # Computing pair-wise distances within synthetic
    dist_ff = metrics.pairwise_distances(fake_sampled, Y=None, metric='minkowski', n_jobs=-1) 
    
    # Removes distances of data points to themselves to avoid 0s within real and synthetic 
    rd_dist_rr = dist_rr[~np.eye(dist_rr.shape[0],dtype=bool)].reshape(dist_rr.shape[0],-1)
    rd_dist_ff = dist_ff[~np.eye(dist_ff.shape[0],dtype=bool)].reshape(dist_ff.shape[0],-1) 
    
    # Computing first and second smallest nearest neighbour distances between real and synthetic
    smallest_two_indexes_rf = [dist_rf[i].argsort()[:2] for i in range(len(dist_rf))]
    smallest_two_rf = [dist_rf[i][smallest_two_indexes_rf[i]] for i in range(len(dist_rf))]       
    # Computing first and second smallest nearest neighbour distances within real
    smallest_two_indexes_rr = [rd_dist_rr[i].argsort()[:2] for i in range(len(rd_dist_rr))]
    smallest_two_rr = [rd_dist_rr[i][smallest_two_indexes_rr[i]] for i in range(len(rd_dist_rr))]
    # Computing first and second smallest nearest neighbour distances within synthetic
    smallest_two_indexes_ff = [rd_dist_ff[i].argsort()[:2] for i in range(len(rd_dist_ff))]
    smallest_two_ff = [rd_dist_ff[i][smallest_two_indexes_ff[i]] for i in range(len(rd_dist_ff))]
    
    # Computing 5th percentiles for DCR and NNDR between and within real and synthetic datasets
    min_dist_rf = np.array([i[0] for i in smallest_two_rf])
    fifth_perc_rf = np.percentile(min_dist_rf,5)
    min_dist_rr = np.array([i[0] for i in smallest_two_rr])
    fifth_perc_rr = np.percentile(min_dist_rr,5)
    min_dist_ff = np.array([i[0] for i in smallest_two_ff])
    fifth_perc_ff = np.percentile(min_dist_ff,5)
    
    return [fifth_perc_rf, fifth_perc_rr, fifth_perc_ff]
#%%
def attribute_disclosure(K, compromised, syndata, attr_compromised, dataset):
    dist = distance_matrix(
        compromised[attr_compromised].to_numpy(),
        syndata[attr_compromised].to_numpy(),
        p=2)
    K_idx = dist.argsort(axis=1)[:, :K]
    
    def most_common(lst):
        return max(set(lst), key=lst.count)
    
    votes = []
    trues = []
    for i in range(len(K_idx)):
        true = np.zeros((len(dataset.categorical_features), ))
        vote = np.zeros((len(dataset.categorical_features), ))
        for j in range(len(dataset.categorical_features)):
            true[j] = compromised.to_numpy()[i, len(dataset.continuous_features) + j]
            vote[j] = most_common(list(syndata.to_numpy()[K_idx[i], len(dataset.continuous_features) + j]))
        votes.append(vote)
        trues.append(true)
    votes = np.vstack(votes)
    trues = np.vstack(trues)
    
    acc = 0
    for j in range(trues.shape[1]):
        acc += (trues[:, j] == votes[:, j]).mean()
    acc /= trues.shape[1]
    return acc

def AttributeDisclosure(train_dataset, syndata):
    """
    Reference:
    [1] A Review of Attribute Disclosure Control
    
    : lower is better
    """
    continuous = train_dataset.continuous_features
    
    train_ = train_dataset.raw_data.copy()
    syndata_ = syndata.copy()
    
    mean = train_[continuous].mean()
    std = train_[continuous].std()
    train_[continuous] -= mean
    train_[continuous] /= std
    syndata_[continuous] -= mean
    syndata_[continuous] /= std
    
    compromised_idx = np.random.choice(
        range(len(train_)), 
        int(len(train_) * 0.01), 
        replace=False)
    compromised = train_.iloc[compromised_idx].reset_index().drop(columns=['index'])
    
    result = []
    for attr_num in [3, 5, 10, 15]:
        if attr_num > len(train_dataset.continuous_features):
            continue
        else:
            attr_compromised = train_dataset.continuous_features[:attr_num]
            for K in [1, 10, 100]:
                acc = attribute_disclosure(
                    K, compromised, syndata_.copy(), attr_compromised, train_dataset)
                result.append(acc)
    AD = np.mean(result)
    return AD
#%%
import numpy as np
from sklearn.metrics.cluster import contingency_matrix

from ..utils import check_adata, check_batch


def purity(adata, label_key, cluster_key):
    check_adata(adata)
    check_batch(cluster_key, adata.obs)
    check_batch(label_key, adata.obs)

    labels_true = adata.obs[label_key].to_numpy()
    labels_pred = adata.obs[cluster_key].to_numpy()

    matrix = contingency_matrix(labels_true, labels_pred)
    return np.sum(np.max(matrix, axis=0)) / np.sum(matrix)


def batch_purity(adata, batch_key, cluster_key):
    check_adata(adata)
    check_batch(cluster_key, adata.obs)
    check_batch(batch_key, adata.obs)

    labels_true = adata.obs[batch_key].to_numpy()
    labels_pred = adata.obs[cluster_key].to_numpy()

    matrix = contingency_matrix(labels_true, labels_pred)
    return 1 - np.sum(np.max(matrix, axis=0)) / np.sum(matrix)

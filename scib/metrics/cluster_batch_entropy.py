"""
Cluster Batch Entropy (CBE)

This metric quantifies how well batches are mixed within clusters. 
It computes the weighted average of batch entropy across all clusters.

Reference formula (from benchmark design):
    CBE = 1 - (sum_k n_k * KL(P_k || P)) / (log(n_batch) * sum_k n_k)

Where:
- n_k: number of cells in cluster k
- P_k: batch distribution within cluster k
- P: overall batch distribution

Status: Implemented
"""

import numpy as np
from scipy.special import rel_entr

def cluster_batch_entropy(adata, label_key, cluster_key):
    """
    Compute Cluster Batch Entropy (CBE) score.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with obs annotations for clustering and batch.
    label_key : str
        Key in `adata.obs` indicating batch labels.
    cluster_key : str
        Key in `adata.obs` indicating clustering labels.

    Returns
    -------
    cbe_score : float
        Cluster batch entropy score (between 0 and 1).
    """
    df = adata.obs[[cluster_key, label_key]].copy()

    # Total number of cells and batches
    n_total = len(df)
    batch_labels = df[label_key].unique()
    n_batches = len(batch_labels)

    # Compute global batch distribution P
    p_global = df[label_key].value_counts(normalize=True).reindex(batch_labels, fill_value=0).values

    kl_sum = 0
    for cluster, subdf in df.groupby(cluster_key):
        n_k = len(subdf)
        p_k = subdf[label_key].value_counts(normalize=True).reindex(batch_labels, fill_value=0).values
        kl_div = np.sum(rel_entr(p_k, p_global))  # KL(P_k || P)
        kl_sum += n_k * kl_div

    cbe = 1 - (kl_sum / (np.log(n_batches) * n_total))
    return cbe

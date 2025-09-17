# scib/metrics/batch_ari.py
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import numpy as np

from ..utils import check_adata, check_batch

def batch_ari(adata, label_key, cluster_key) -> float:
    """Batch ARI

    ARI between cluster assignments and BATCH labels.

    Parameters
    ----------
    adata : AnnData
    label_key : str
        obs column containing batch labels.
    cluster_key : str
        obs column containing cluster labels.
    Returns
    -------
    float
        Adjusted Rand Index in [-1, 1].
    """
    check_adata(adata)
    check_batch(cluster_key, adata.obs)
    check_batch(label_key, adata.obs)

    clusters = adata.obs[cluster_key]
    batches = adata.obs[label_key]

    # robust to dtype issues
    if not pd.api.types.is_categorical_dtype(clusters):
        clusters = clusters.astype("category")
    if not pd.api.types.is_categorical_dtype(batches):
        batches = batches.astype("category")

    y_pred = clusters.cat.codes.to_numpy()
    y_true = batches.cat.codes.to_numpy()

    # guard empty / constant cases
    if y_pred.size == 0 or y_true.size == 0:
        return np.nan
    if len(np.unique(y_pred)) <= 1 or len(np.unique(y_true)) <= 1:
        return 0.0

    # The lower the ARI, the better
    return 1 - adjusted_rand_score(y_true, y_pred)

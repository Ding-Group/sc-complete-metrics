import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from ..utils import check_adata, check_batch


def batch_ari(adata, label_key, cluster_key) -> float:
    """Batch ARI

    Computes the adjusted Rand index between clustering assignments 
    and batch labels instead of cell type labels. A low ARI indicates 
    that batch labels are well mixed across clusters, so the score is 
    reported as ``1 - ARI``.

    :param adata: AnnData object containing the data and annotations.
    :param label_key: Key in adata.obs for batch labels.
    :param cluster_key: Key in adata.obs for cluster assignments.
    :return: One minus the adjusted Rand index. Returns 0.0 if there is only one cluster or one batch.
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

    if len(np.unique(y_pred)) <= 1 or len(np.unique(y_true)) <= 1:
        return 0.0

    # The lower the ARI, the better
    return 1 - adjusted_rand_score(y_true, y_pred)

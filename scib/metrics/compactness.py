import numpy as np
from sklearn.metrics import pairwise_distances

from ..utils import check_adata, check_batch


def compactness(adata, label_key, cluster_key, embed) -> float:
    """Compactness (COMPACT)
    
    Computes the average intra-cluster distance.

    :param adata: AnnData object
    :param label_key: (unused) Column in adata.obs with true labels
    :param cluster_key: Column in adata.obs with cluster labels
    :param embed: Key in adata.obsm containing the embedding
    :return: Compactness score
    """
    check_adata(adata)
    check_batch(cluster_key, adata.obs)

    X = adata.obsm[embed]
    labels = adata.obs[cluster_key].to_numpy()

    # compute each avg
    compactness_scores = []
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        n = len(idx)
        if n < 2:
            continue

        Xi = X[idx]

        # Accumulate pairwise distances without constructing full matrix
        total = 0.0
        count = 0
        for i in range(n - 1):
            # pairwise distances from one point to all later points
            d = pairwise_distances(Xi[i : i + 1], Xi[i + 1 :], metric="euclidean")
            total += d.sum()
            count += d.size

        compactness_scores.append(total / count)

    # Invert to make higher values better
    return -np.mean(compactness_scores)

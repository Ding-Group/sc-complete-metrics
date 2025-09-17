import numpy as np
from sklearn.metrics import pairwise_distances
from ..utils import check_adata, check_batch


def compactness(adata, label_key, cluster_key, embed):
    """
    Compute the average intra-cluster distance (compactness).

    :param adata: AnnData object
    :param label_key: (unused) Column in adata.obs with true labels (for signature compatibility)
    :param cluster_key: Column in adata.obs with cluster labels
    :param embed: Key in adata.obsm containing the embedding
    :return: Average intra-cluster distance
    """
    check_adata(adata)
    check_batch(cluster_key, adata.obs)

    X = adata.obsm[embed]
    labels = adata.obs[cluster_key].to_numpy()

    # compute each avg
    compactness_scores = []
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        if len(idx) < 2:
            continue
        dists = pairwise_distances(X[idx])
        upper = dists[np.triu_indices_from(dists, k=1)]
        compactness_scores.append(upper.mean())

    # Invert to make higher values better
    return -np.mean(compactness_scores)

import numpy as np
from sklearn.metrics import pairwise_distances
from ..utils import check_adata, check_batch

def separation(
    adata,
    label_key,
    cluster_key,
    embed
) -> float:
    """Separation

    Compute the average inter-cluster distance (separation).

    :param adata: AnnData object
    :param cluster_key: Column in adata.obs with cluster labels
    :param label_key: Column in adata.obs with cell type labels
    :param embed: Key in adata.obsm containing the embedding
    :return: Separation score
    """
    check_adata(adata)
    check_batch(cluster_key, adata.obs)

    X = adata.obsm[embed]
    labels = adata.obs[cluster_key].to_numpy()
    unique_labels = np.unique(labels)

    # Compute centroids for each cell type
    centroids = []
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        if idx.size == 0:
            continue
        centroids.append(X[idx].mean(axis=0))
    centroids = np.array(centroids)

    # If fewer than two clusters, return 0
    if centroids.shape[0] < 2:
        return 0.0

    # Compute pairwise distances between centroids
    dists = pairwise_distances(centroids)
    upper = dists[np.triu_indices_from(dists, k=1)]

    # Invert to make larger values better
    return -float(np.mean(upper))

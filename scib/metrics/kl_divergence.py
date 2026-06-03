import numpy as np
import scipy.stats
from sklearn.neighbors import NearestNeighbors
from ..utils import check_adata, check_batch


def kl_divergence(
    adata,
    batch_key,
    embed,
    k: int = None,
    metric: str = "euclidean"
):
    """Compute average KL divergence between local and global batch distribution
    in nearest neighborhoods.

    :param adata: AnnData object
    :param batch_key: Column in ``adata.obs`` with batch annotations.
    :param k: Number of neighbors for local distribution. Default to 5 x # of batches
    :param metric: Distance metric for nearest neighbor computation
    """
    # Sanity checks
    check_adata(adata)
    check_batch(batch_key, adata.obs)

    X = adata.obsm[embed]

    # Get batch labels
    batch_labels = adata.obs[batch_key].to_numpy()
    unique_batches, counts = np.unique(batch_labels, return_counts=True)
    n_cells = len(batch_labels)

    if k is None:
        k = 5 * len(unique_batches)

    # Global batch distribution
    global_dist = counts / n_cells

    # Fit kNN and get neighbors (excluding self)
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    knn_indices = nn.kneighbors(X, return_distance=False)[:, 1:]

    # Compute KL for each cell
    kl_list = []
    for i in range(n_cells):
        local = batch_labels[knn_indices[i]]
        local_counts = np.array([np.sum(local == b) for b in unique_batches])
        local_dist = local_counts / np.sum(local_counts)
        # add tiny epsilon to avoid zeros
        kl = scipy.stats.entropy(local_dist + 1e-10, global_dist + 1e-10)
        kl_list.append(kl)

    return float(np.mean(kl_list))

import numpy as np
from scipy.stats import entropy

from ..utils import check_adata, check_batch


def batch_shannon_entropy(
    adata,
    batch_key: str,
    cluster_key: str,
    normalize: bool = True
) -> float:
    """
    Weighted average Shannon entropy of batch labels within each cluster.
    Higher = better batch mixing.

    If normalize=True, returns a value in [0, 1] by dividing by log(#batches).
    """
    check_adata(adata)
    check_batch(cluster_key, adata.obs)
    check_batch(batch_key, adata.obs)

    N = adata.n_obs
    batches = adata.obs[batch_key].astype("category").cat.codes
    clusters = adata.obs[cluster_key].astype("category").cat.codes

    max_ent = np.log(len(np.unique(batches))) if normalize and len(batches) > 1 else 1.0

    score = 0.0
    for c in np.unique(clusters):
        n_c = np.sum(clusters == c)
        sub = batches[clusters == c]
        _, counts = np.unique(sub, return_counts=True)
        freqs = counts / counts.sum()
        h = entropy(freqs)
        score += (n_c / N) * h

    if normalize and max_ent > 0:
        score /= max_ent
    return float(score)


def celltype_shannon_entropy(
    adata,
    label_key: str,
    cluster_key: str,
    normalize: bool = True
) -> float:
    """
    Weighted average Shannon entropy of cell type labels within each cluster.

    If normalize=True, returns a value in [0, 1] by dividing by log(#batches).
    """
    check_adata(adata)
    check_batch(cluster_key, adata.obs)
    check_batch(label_key, adata.obs)

    N = adata.n_obs
    labels = adata.obs[label_key].astype("category").cat.codes
    clusters = adata.obs[cluster_key].astype("category").cat.codes

    max_ent = np.log(len(np.unique(labels))) if normalize and len(labels) > 1 else 1.0

    score = 0.0
    for c in np.unique(clusters):
        n_c = np.sum(clusters == c)
        sub = labels[clusters == c]
        _, counts = np.unique(sub, return_counts=True)
        freqs = counts / counts.sum()
        h = entropy(freqs)
        score += (n_c / N) * h

    if normalize and max_ent > 0:
        score /= max_ent
    if normalize:
        # The lower the entropy the better, so flip the sign
        return 1 - float(score)
    else:
        return -float(score)

import numpy as np
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors

from ..utils import check_adata, check_batch


def shannon_entropy_nn(adata, batch_key, label_key, embed, n_neighbors=50):
    """Shannon entropy

    Compute the batch and cell type Shannon entropy within each cell's
    local neighbourhood.

    :param adata: AnnData object
    :param batch_key: Key in adata.obs with batch labels
    :param label_key: Key in adata.obs with cell type labels
    :param embed: Key in adata.obsm with integrated embedding,
    :param n_neighbors: Number of nearest neighbors. Default: 50
    :return: Shannon entropy
    """
    # Basic checks
    check_adata(adata)
    check_batch(batch_key, adata.obs)
    check_batch(label_key, adata.obs)

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(adata.obsm[embed])
    _, indices = nn.kneighbors(adata.obsm[embed])
    indices = indices[:, 1:]

    batches = adata.obs[batch_key].astype("category").cat.codes
    labels = adata.obs[label_key].astype("category").cat.codes
    n_batches, n_labels = len(np.unique(batches)), len(np.unique(labels))

    batch_entropy = []
    celltype_entropy = []

    for ind in indices:
        selected_batches = batches[ind]
        selected_labels = labels[ind]

        batch_entropy.append(entropy_(selected_batches, n_batches))
        celltype_entropy.append(entropy_(selected_labels, n_labels))

    # The lower the celltype entropy, the better
    return np.mean(batch_entropy), 1 - np.mean(celltype_entropy)


def entropy_(selected_labels, n_labels):
    """
    Shannon entropy calculation normalized by log(n_labels).

    :param labels: Labels for the group.
    :param n_labels: Number of unique batches.
    """
    # Compute frequency distribution
    _, counts = np.unique(selected_labels, return_counts=True)
    freqs = counts / counts.sum()

    # Shannon entropy (base = natural log, matches R log())
    h = entropy(freqs, base=2)

    # Normalize by log(n_labels)
    return h / np.log2(n_labels)

import numpy as np
import scanpy as sc
from scipy.stats import entropy

from ..utils import check_adata, check_batch


def shannon_entropy_nn(adata, batch_key, label_key, k=None, embed=None):
    """Shannon entropy

    Compute the batch and cell type Shannon entropy within each cell's
    local neighbourhood.

    :param adata: AnnData object
    :param batch_key: Key in adata.obs with batch labels
    :param label_key: Key in adata.obs with cell type labels
    :param k: Number of nearest neighbors. If None, default to graph computed
        in adata.obsp['distances'].
    :param embed: Key in adata.obsm with integrated embedding, to be used if
        k provided.
    :return: Shannon entropy
    """
    # Basic checks
    check_adata(adata)
    check_batch(batch_key, adata.obs)
    check_batch(label_key, adata.obs)

    graph_key = 'distances'
    if k is not None:
        sc.pp.neighbors(adata, n_neighbors=k, use_rep=embed, key_added='shannon_distances')
        graph_key = 'shannon_distances'

    batches = adata.obs[batch_key].astype("category").cat.codes
    labels = adata.obs[label_key].astype("category").cat.codes
    n_batches, n_labels = len(np.unique(batches)), len(np.unique(labels))

    batch_entropy = []
    celltype_entropy = []
    knn_graph = adata.obsp[graph_key]

    for i in range(knn_graph.shape[0]):
        row = knn_graph.getrow(i)
        indices = row.indices
        indices = np.where(indices >= 1)[0]  # TODO: This line might be wrong
        selected_batches = batches[indices]
        selected_labels = labels[indices]

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
    h = entropy(freqs, base=np.e)

    # Normalize by log(n_labels)
    return h / np.log(n_labels)

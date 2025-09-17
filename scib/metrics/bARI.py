# Source: https://github.com/theislab/atlas-feature-selection-benchmark/blob/main/bin/metric-bARI.py
from ..utils import check_adata, check_batch
from .clustering import cluster_optimal_resolution


def bARI(adata, label_key, cluster_key):
    """
    Balanced Adjusted Rand Index (bARI) for an AnnData object.
    A wrapper around `balanced_clustering.balanced_adjusted_rand_index()` with the interface expected by `scib.metrics.cluster_optimal_resolution`.

    Parameters
    ----------
    adata
        Anndata object with cluster assignments
    cluster_key
        Name of the column in adata.obs containing cluster assignments
    label_key
        Name of the column in adata.obs containing ground truth labels

    Returns
    -------
    The bARI score [0, 1] where 0 is no match between the pair of labels, and 1 a perfect match.
    """

    from balanced_clustering import balanced_adjusted_rand_index

    clusters = adata.obs[cluster_key].to_numpy()
    labels = adata.obs[label_key].to_numpy()

    return balanced_adjusted_rand_index(labels_true=labels, labels_pred=clusters)

# Source: https://github.com/theislab/atlas-feature-selection-benchmark/blob/main/bin/metric-bARI.py
from ..utils import check_adata, check_batch


def bv_measure(adata, label_key, cluster_key):
    """
    Balanced V-measure for an AnnData object.

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

    from balanced_clustering import balanced_homogeneity_completeness_v_measure

    clusters = adata.obs[cluster_key].to_numpy()
    labels = adata.obs[label_key].to_numpy()

    _, _, vm = balanced_homogeneity_completeness_v_measure(labels_true=labels, labels_pred=clusters)
    return vm

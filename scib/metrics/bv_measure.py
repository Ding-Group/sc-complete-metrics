import numpy as np


def bv_measure(adata, label_key, cluster_key) -> float:
    """Balanced V-measure

    Computes the balanced V-measure between clustering assignments
    and true labels from Maan et al., 2024 "Characterizing
    the impacts of dataset imbalance on single-cell data integration".

    doi:10.1038/s41587-023-02097-9.

    This function wraps around ``balanced_clustering.balanced_v_measure()``.

    :param adata: AnnData object containing the data and annotations.
    :param label_key: Key in adata.obs for true cell type labels.
    :param cluster_key: Key in adata.obs for cluster assignments.
    :return: Balanced V-measure score. Returns NaN if there is only one cluster.
    """

    from balanced_clustering import balanced_v_measure

    clusters = adata.obs[cluster_key].to_numpy()
    labels = adata.obs[label_key].to_numpy()

    if len(np.unique(clusters)) == 1:
        return np.nan

    return balanced_v_measure(labels_true=labels, labels_pred=clusters)

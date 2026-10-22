# Source: https://github.com/theislab/atlas-feature-selection-benchmark/blob/main/bin/metric-bARI.py
def bARI(adata, label_key, cluster_key) -> float:
    """Balanced Adjusted Rand Index (BARI)

    Computes the balanced adjusted Rand index between clustering assignments 
    and true labels from Maan et al., 2024  "Characterizing the impacts of 
    dataset imbalance on single-cell data integration".

    doi:10.1038/s41587-023-02097-9

    This function wraps around ``balanced_clustering.balanced_adjusted_rand_index()``
    with the interface expected by :func:`scib.metrics.cluster_optimal_resolution`.

    :param adata: AnnData object containing the data and annotations.
    :param label_key: Key in adata.obs for true cell type labels.
    :param cluster_key: Key in adata.obs for cluster assignments.
    :return: Balanced adjusted Rand index score
    """

    from balanced_clustering import balanced_adjusted_rand_index

    clusters = adata.obs[cluster_key].to_numpy()
    labels = adata.obs[label_key].to_numpy()

    return balanced_adjusted_rand_index(labels_true=labels, labels_pred=clusters)

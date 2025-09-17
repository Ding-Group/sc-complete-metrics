from sklearn.metrics import adjusted_mutual_info_score
from ..utils import check_adata, check_batch


def ami(adata, label_key, cluster_key, average_method='arithmetic'):
    """Adjusted Mutual Information (AMI)

    Computes the adjusted mutual information between clustering assignments and true labels.

    :param adata: AnnData object containing the data and annotations.
    :param cluster_key: Key in adata.obs for cluster assignments.
    :param label_key: Key in adata.obs for true cell type labels.
    :param average_method: String specifying the averaging method to use.
        Must be one of 'arithmetic' (default), 'min', 'geometric', or 'max'.
    :return: Adjusted mutual information score (float).
    """
    check_adata(adata)
    check_batch(cluster_key, adata.obs)
    check_batch(label_key, adata.obs)

    clusters = adata.obs[cluster_key].tolist()
    labels = adata.obs[label_key].tolist()

    if len(clusters) != len(labels):
        raise ValueError(
            f"Length mismatch: '{cluster_key}' has {len(clusters)} entries, "
            f"but '{label_key}' has {len(labels)} entries."
        )

    # Compute and return AMI
    return adjusted_mutual_info_score(
        clusters,
        labels,
        average_method=average_method
    )

from sklearn.metrics import v_measure_score

from ..utils import check_adata, check_batch


def v_measure(adata, label_key, cluster_key) -> float:
    """V-measure (VM)

    Computes the V-measure between clustering assignments and true labels.
    The V-measure is the harmonic mean of homogeneity and completeness, and ranges from 0 to 1.

    :param adata: AnnData object containing the data and annotations.
    :param label_key: Key in ``adata.obs`` for groundtruth cell type labels.
    :param cluster_key: Key in ``adata.obs`` for cluster assignments.
    :return: V-measure score
    """
    check_adata(adata)
    check_batch(cluster_key, adata.obs)
    check_batch(label_key, adata.obs)

    labels_true = adata.obs[label_key].to_numpy()
    labels_pred = adata.obs[cluster_key].to_numpy()

    return v_measure_score(labels_true, labels_pred)

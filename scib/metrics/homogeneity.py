from sklearn.metrics import homogeneity_score

from ..utils import check_adata, check_batch


def homogeneity(adata, label_key, cluster_key) -> float:
    """Homogeneity score (HOMO)

    Computes the homogeneity score.

    :param adata: Integrated AnnData object
    :param label_key: Key in ``adata.obs`` with groundtruth labels
    :param cluster_key: Key in ``adata.obs`` with cluster assignments
    :return: Homogeneity score
    """
    check_adata(adata)
    check_batch(cluster_key, adata.obs)
    check_batch(label_key, adata.obs)

    labels_true = adata.obs[label_key].to_numpy()
    labels_pred = adata.obs[cluster_key].to_numpy()

    return homogeneity_score(labels_true, labels_pred)

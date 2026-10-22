from sklearn.cluster import KMeans
from sklearn.metrics import matthews_corrcoef as _sklearn_mcc

from ..utils import check_adata, check_batch


def matthews_corrcoef(adata, label_key, cluster_key) -> float:
    """Matthews correlation coefficient

    Compute the Matthews correlation coefficient (MCC) between predicted clusters and true labels.

    :param adata: AnnData object.
    :param label_key: Key in ``adata.obs`` for groundtruth cell type labels.
    :param cluster_key: Key in ``adata.obs`` for cluster assignments.
    :return: Matthews correlation cofficient.
    """
    check_adata(adata)
    check_batch(cluster_key, adata.obs)
    check_batch(label_key, adata.obs)

    labels_true = adata.obs[label_key].to_numpy()
    mapping = adata.obs.groupby(cluster_key)[label_key].agg(lambda x: x.mode().iloc[0])
    labels_pred = adata.obs[cluster_key].map(mapping)

    return float(_sklearn_mcc(labels_true, labels_pred))

# scib_extension/scib/metrics/matthews_corrcoef.py

from sklearn.metrics import matthews_corrcoef as _sklearn_mcc
from ..utils import check_adata, check_batch

def matthews_corrcoef(adata, label_key, cluster_key):
    """
    Compute the Matthews correlation coefficient (MCC) between predicted clusters and true labels.
    """
    check_adata(adata)
    check_batch(cluster_key, adata.obs)
    check_batch(label_key, adata.obs)

    labels_true = adata.obs[label_key].to_numpy()
    labels_pred = adata.obs[cluster_key].to_numpy()
    return float(_sklearn_mcc(labels_true, labels_pred))

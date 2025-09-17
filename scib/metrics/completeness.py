from sklearn.metrics import completeness_score

from ..utils import check_adata, check_batch


def completeness(adata, cluster_key, label_key):
    check_adata(adata)
    check_batch(cluster_key, adata.obs)
    check_batch(label_key, adata.obs)

    labels_true = adata.obs[label_key].to_numpy()
    labels_pred = adata.obs[cluster_key].to_numpy()

    return completeness_score(labels_true, labels_pred)

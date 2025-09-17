from sklearn.metrics import fowlkes_mallows_score
from ..utils import check_adata, check_batch

def fowlkes_mallows(adata, label_key, cluster_key):
    check_adata(adata)
    check_batch(cluster_key, adata.obs)
    check_batch(label_key, adata.obs)

    y_true = adata.obs[label_key].to_numpy()
    y_pred = adata.obs[cluster_key].to_numpy()
    return fowlkes_mallows_score(y_true, y_pred)

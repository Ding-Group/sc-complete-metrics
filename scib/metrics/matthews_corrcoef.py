# scib_extension/scib/metrics/matthews_corrcoef.py

from sklearn.cluster import KMeans
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
    mapping = adata.obs.groupby(cluster_key)[label_key].agg(lambda x: x.mode().iloc[0])
    labels_pred = adata.obs[cluster_key].map(mapping)

    return float(_sklearn_mcc(labels_true, labels_pred))

def mcc_kmeans(adata, label_key, embed, key_added='kmeans'):
    """Version of Jaccard where number of clusters fixed by KMeans"""
    kmeans = KMeans(n_clusters=adata.obs[label_key].nunique(), random_state=0).fit(adata.obsm[embed])
    adata.obs[key_added] = kmeans

    return float(_sklearn_mcc(adata.obs[label_key], adata.obs[key_added]))

from sklearn.metrics import calinski_harabasz_score
from ..utils import check_adata, check_batch
import numpy as np

def calinski_harabasz(adata, label_key, cluster_key, embed):
    """
    Compute the Calinski-Harabasz index for cluster evaluation.

    A higher Calinski-Harabasz score relates to a model with better defined clusters.

    :param adata: AnnData object
    :param label_key: Unused, kept for consistency with clustering function
    :param cluster_key: Column in adata.obs with cluster labels
    :param embed: Key in adata.obsm containing the embedding
    :return: Calinski-Harabasz index (float)
    """
    check_adata(adata)
    check_batch(cluster_key, adata.obs)

    X = adata.obsm[embed]

    if adata.obs[cluster_key].nunique() == 1:
        return np.nan
    clusters = adata.obs[cluster_key].to_numpy()
    return calinski_harabasz_score(X, clusters)

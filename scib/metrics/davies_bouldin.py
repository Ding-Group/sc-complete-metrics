from sklearn.metrics import davies_bouldin_score
import numpy as np

def davies_bouldin(adata, label_key, cluster_key, embed):
    """Davies-Bouldin Score
    
    :param adata: AnnData object containing the data and annotations.
    :param label_key: Unused parameter.
    :param embed: Key in adata.obsm with integrated embeddings.
    """
    X = adata.obsm[embed]

    if adata.obs[cluster_key].nunique() == 1:
        return np.inf

    # Reverse sign since lower scores mean better clustering.
    return -davies_bouldin_score(X, adata.obs[cluster_key])

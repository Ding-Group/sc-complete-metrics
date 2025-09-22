from sklearn.metrics import davies_bouldin_score


def davies_bouldin(adata, label_key, embed):
    """Davies-Bouldin Score
    
    :param adata: AnnData object containing the data and annotations.
    :param label_key: Unused parameter.
    :param embed: Key in adata.obsm with integrated embeddings.
    """
    X = adata.obsm[embed]

    # Reverse sign since lower scores mean better clustering.
    return -davies_bouldin_score(X, adata.obs[label_key])

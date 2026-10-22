import numpy as np
from sklearn.metrics import davies_bouldin_score


def davies_bouldin(adata, label_key, cluster_key, embed) -> float:
    """Davies-Bouldin index (CLUSTDBI, CDBI)

    Computes the Davies-Bouldin index.

    :param adata: AnnData object containing the data and annotations.
    :param label_key: Unused parameter.
    :param cluster_key: Key in ``adata.obs`` with cluster assignments.
    :param embed: Key in adata.obsm with integrated embeddings.
    :return: Davies-Bouldin index
    """
    X = adata.obsm[embed]

    if adata.obs[cluster_key].nunique() == 1:
        return np.inf

    # Reverse sign since lower scores mean better clustering.
    return -davies_bouldin_score(X, adata.obs[cluster_key])

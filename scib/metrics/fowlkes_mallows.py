from sklearn.metrics import fowlkes_mallows_score

from ..utils import check_adata, check_batch


def fowlkes_mallows(adata, label_key, cluster_key) -> float:
    """Fowlkes-Mallows score.

    :param adata: AnnData object.
    :param label_key: Key in adata.obs containing cell type information.
    :param cluster_key: Key in adata.obs containing cluster information.
    :return: Fowlkes-Mallows score.
    """
    check_adata(adata)
    check_batch(cluster_key, adata.obs)
    check_batch(label_key, adata.obs)

    y_true = adata.obs[label_key].to_numpy()
    y_pred = adata.obs[cluster_key].to_numpy()
    return fowlkes_mallows_score(y_true, y_pred)

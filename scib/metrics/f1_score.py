from sklearn.metrics.cluster import pair_confusion_matrix

from ..utils import check_adata, check_batch


def f1_score_macro(adata, label_key, cluster_key):
    """
    Compute the macro-averaged F1 score between predicted clusters and ground truth labels.

    Parameters:
    - adata: AnnData object
    - cluster_key: str, column in adata.obs with predicted cluster labels
    - label_key: str, column in adata.obs with ground truth labels

    Returns:
    - Macro F1 score (float)
    """
    check_adata(adata)
    check_batch(cluster_key, adata.obs)
    check_batch(label_key, adata.obs)

    labels_true = adata.obs[label_key].to_numpy()
    labels_pred = adata.obs[cluster_key].to_numpy()

    conf_mat = pair_confusion_matrix(labels_true, labels_pred)
    tp = conf_mat[1, 1]
    fp = conf_mat[0, 1]
    fn = conf_mat[1, 0]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)

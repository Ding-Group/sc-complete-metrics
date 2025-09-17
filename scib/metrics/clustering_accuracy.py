import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from ..utils import check_adata, check_batch


# TODO: Should this be a one-to-one matching between clusters to labels,
# or should we have multiple clusters map to one label?

def clustering_accuracy(adata, label_key, cluster_key):
    """
    Compute clustering accuracy by matching predicted clusters to true labels.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing cluster and label annotations in adata.obs.
    label_key : str
        Key in adata.obs for ground truth labels.
    cluster_key : str
        Key in adata.obs for predicted cluster labels.

    Returns
    -------
    float
        Clustering accuracy after optimal one-to-one label matching.
    """
    # Validate input AnnData and keys
    check_adata(adata)
    check_batch(cluster_key, adata.obs)
    check_batch(label_key, adata.obs)

    # Convert true labels to integer codes
    true_series = adata.obs[label_key]
    if not pd.api.types.is_categorical_dtype(true_series):
        true_series = true_series.astype("category")
    labels_true = true_series.cat.codes.to_numpy()

    # Convert predicted clusters to integer codes
    pred_series = adata.obs[cluster_key]
    if not pd.api.types.is_categorical_dtype(pred_series):
        pred_series = pred_series.astype("category")
    labels_pred = pred_series.cat.codes.to_numpy()

    # Build contingency matrix between predicted and true labels
    n_classes = max(labels_pred.max(), labels_true.max()) + 1
    contingency = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i in range(labels_pred.shape[0]):
        contingency[labels_pred[i], labels_true[i]] += 1

    # Solve assignment problem (Hungarian) for best label matching
    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)

    # Compute accuracy as sum of matches over total cells
    accuracy = contingency[row_ind, col_ind].sum() / labels_pred.shape[0]
    return float(accuracy)

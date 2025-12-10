# Based on: https://github.com/theislab/atlas-feature-selection-benchmark/blob/main/bin/metric-bARI.py
from ..utils import check_adata, check_batch
import numpy as np

def bAMI(adata, label_key, cluster_key):
    """
    Balanced Adjusted Mutual Information (bAMI) for an AnnData object.

    Parameters
    ----------
    adata
        Anndata object with cluster assignments
    cluster_key
        Name of the column in adata.obs containing cluster assignments
    label_key
        Name of the column in adata.obs containing ground truth labels

    Returns
    -------
    The bAMI score [0, 1] where 0 is no match between the pair of labels, and 1 a perfect match.
    """

    from balanced_clustering import balanced_adjusted_mutual_info

    clusters = adata.obs[cluster_key].to_numpy()
    labels = adata.obs[label_key].to_numpy()

    if len(np.unique(clusters)) == 1:
        # Must have at least two clusters.
        return np.nan

    return balanced_adjusted_mutual_info(labels_true=labels, labels_pred=clusters)

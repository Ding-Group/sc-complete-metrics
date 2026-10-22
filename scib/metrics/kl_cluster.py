import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix


def compute_cooccurrence_matrix(labels, batches):
    mat = np.zeros((labels.max() + 1, batches.max() + 1), dtype=int)
    for k, b in zip(labels, batches):
        mat[k, b] += 1
    return mat


def kl_cluster(adata, batch_key, cluster_key, label_key) -> float:
    """KL cluster (KLC)

    Assumes that the number of clusters equals the number of labels.
    Because of this constraint, it is not analyzed in our metric evaluation.

    :param adata: AnnData object.
    :param batch_key: Key in adata.obs with batch information.
    :param cluster_key: Key in adata.obs with cluster information.
    :param label_key: Key in adata.obs with cell type information.
    :return: The cluster KL divergence score
    """
    labels = adata.obs[label_key].astype('category').cat.codes
    clusters = adata.obs[cluster_key].astype('category').cat.codes
    batches = adata.obs[batch_key].astype('category').cat.codes

    if labels.max() != clusters.max():
        raise ValueError(
            "KL divergence cluster expects the number of"
            " clusters to equal the number of cell types."
        )
    
    # Align clusters with labels
    contingency = confusion_matrix(labels, clusters)
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = dict(zip(col_ind, row_ind))
    clusters_aligned = np.array([mapping[label] for label in clusters])

    O = compute_cooccurrence_matrix(clusters_aligned, batches)
    E = compute_cooccurrence_matrix(labels, batches)

    BER = np.sum(O * np.log((O + 1) / (E + 1)))

    # Change to higher is better.
    return -BER

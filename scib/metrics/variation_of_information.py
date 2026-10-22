import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score


def variation_of_information(adata, label_key, cluster_key, normalize=True) -> float:
    """Variation of information

    :param adata: AnnData object.
    :param label_key: Key in adata.obs containing cell type labels.
    :param cluster_key: Key in adata.obs containing cluster labels.
    :param normalize: Normalize to [0, 1].
    :return: Variation of information score
    """
    labels_true = adata.obs[label_key].astype('category').cat.codes
    labels_pred = adata.obs[cluster_key].astype('category').cat.codes

    contingency = np.histogram2d(
        labels_true, labels_pred, bins=(np.unique(labels_true).size, np.unique(labels_pred).size)
    )[0]

    Pa = contingency.sum(axis=1) / adata.n_obs
    Pb = contingency.sum(axis=0) / adata.n_obs

    H1 = entropy(Pa, base=np.e)
    H2 = entropy(Pb, base=np.e)

    # I(C, C')
    I = mutual_info_score(labels_true, labels_pred)

    if normalize:
        return 1 - (H1 + H2 - 2 * I) / np.log(adata.n_obs)
    return -(H1 + H2 - 2 * I)

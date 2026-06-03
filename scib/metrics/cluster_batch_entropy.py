import numpy as np
from scipy.special import rel_entr

def cluster_batch_entropy(adata, label_key, cluster_key):
    """Cluster batch entropy

    :param adata: AnnData object with obs annotations for clustering and batch
    :param label_key: Key in ``adata.obs`` indicating batch labels
    :param cluster_key: Key in ``adata.obs`` indicating clustering labels
    :return: Cluster batch entropy score (between 0 and 1)
    """
    df = adata.obs[[cluster_key, label_key]].copy()

    n_total = len(df)
    batch_labels = df[label_key].unique()
    n_batches = len(batch_labels)

    p_global = df[label_key].value_counts(normalize=True).reindex(batch_labels, fill_value=0).values

    kl_sum = 0
    for cluster, subdf in df.groupby(cluster_key):
        n_k = len(subdf)
        p_k = subdf[label_key].value_counts(normalize=True).reindex(batch_labels, fill_value=0).values
        kl_div = np.sum(rel_entr(p_k, p_global))  # KL(P_k || P)
        kl_sum += n_k * kl_div

    cbe = 1 - (kl_sum / (np.log(n_batches) * n_total))
    return cbe

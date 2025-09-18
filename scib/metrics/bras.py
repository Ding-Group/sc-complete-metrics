from scib_metrics.metrics import bras as bras_


def bras(
    adata,
    label_key,
    batch_key,
    embed,
    chunk_size=256,
    metric = "cosine",
    between_cluster_distances = "mean_other"
):
    """BRAS

    Wrapper around BRAS from scib-metrics, as defined in
    Rautenstrauch & Ohler.
    """
    X = adata.obsm[embed]
    labels = adata.obs[label_key]
    batch = adata.obs[batch_key]

    return bras_(X, labels, batch, chunk_size, metric, between_cluster_distances)

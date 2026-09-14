def bras(
    adata,
    label_key,
    batch_key,
    embed,
    chunk_size: int = 256,
    metric: str = "cosine",
    between_cluster_distances: str = "mean_other"
):
    from scib_metrics import bras as bras_
    """BRAS.

    Wrapper around BRAS from scib-metrics, as defined in
    Rautenstrauch & Ohler.

    :param adata: AnnData object.
    :param label_key: Key in adata.obs containing cell type information.
    :param batch_key: Key in adata.obs containing batch information.
    :param embed: Key in adata.obsm containing the embedding.
    :param chunk_size: Chunk size for BRAS computation.
    :param metric: Distance metric for BRAS computation.
    :param between_cluster_distances: Method for computing between-cluster distances.
    :return: BRAS score.
    """
    X = adata.obsm[embed]
    labels = adata.obs[label_key]
    batch = adata.obs[batch_key]

    return bras_(X, labels, batch, chunk_size, metric, between_cluster_distances)

import sklearn


def pstdb(
    adata,
    batch_key,
    label_key,
    embed,
    k: int = 50,
    metric: str = "euclidean"
):
    """Proportion of nearest neighbors of the same cell type but a different batch.

    :param adata: AnnData object
    :param batch_key: Column in ``adata.obs`` with batch information
    :param label_key: Column in ``adata.obs`` with cell type information
    :param embed: Key in ``adata.obsm`` with integrated embedding
    :param k: Number of nearest neighbors
    :param metric: Distance metric for nearest neighbor computation
    """
    nn = sklearn.neighbors.NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(adata.obsm[embed])
    nn.fit(adata.obsm[embed])
    nni = nn.kneighbors(adata.obsm[embed], return_distance=False)

    batches = adata.obs[batch_key].to_numpy()
    labels = adata.obs[label_key].to_numpy()
    nn_batches = batches[nni[:, 1:]]
    nn_labels = labels[nni[:, 1:]]

    cell_batches = batches[nni[:, 0]]
    cell_labels = labels[nni[:, 0]]

    batch_mismatch = nn_batches != cell_batches[:, None]
    label_match = nn_labels == cell_labels[:, None]

    return (batch_mismatch & label_match).mean()

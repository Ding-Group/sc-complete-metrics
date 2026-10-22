import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..utils import check_adata, check_batch


def alignment(
    adata_int,
    batch_key: str,
    embed: str,
    n_neighbors: int = None,
    metric: str = "euclidean",
    seed: int = 0,
) -> float:
    """Alignment score

    Computes the alignment score from Butler et al. 2018 "Integrating single-cell
    transcriptomic data across different conditions, technologies, and species".

    doi:10.1038/nbt.4096

    :param adata_int: AnnData with integrated embedding
    :param batch_key: Column in adata.obs with batch information
    :param embed: Key in adata.obsm with integrated embedding
    :param metric: Distance metric for nearest neighbor computation
    :param n_neighbors: Number of nearest neighbors to use.
        Defaults to 1% of the number of cells.
    :param seed: Seed for random downsampling.
    :return: The alignment score
    """
    check_adata(adata_int)
    check_batch(batch_key, adata_int.obs)

    # Downsample the data to the size of the smallest batch
    smallest_batch = adata_int.obs[batch_key].value_counts().min()
    sampled_indices = (
        adata_int.obs.groupby(batch_key)
        .sample(n=smallest_batch, random_state=seed)
        .index
    )
    adata_int = adata_int[sampled_indices].copy()

    X = np.asarray(adata_int.obsm[embed])
    batches = adata_int.obs[batch_key].to_numpy()
    n_batches = len(adata_int.obs[batch_key].unique())

    if n_neighbors is None:
        n_neighbors = adata_int.n_obs // 100
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric, n_jobs=-1).fit(X)
    knn_graph = nn.kneighbors(X, return_distance=False)[:, 1:]
    scores = []
    for i, neighbors in enumerate(knn_graph):
        same = np.sum(batches[neighbors] == batches[i])
        scores.append(same)

    x_ = np.mean(scores)
    return 1 - (x_ - n_neighbors / n_batches) / (n_neighbors - n_neighbors / n_batches)

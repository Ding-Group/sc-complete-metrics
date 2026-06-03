import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors


def mixing(adata, batch_key, embed, k: int = 5, max_k: int = 300, metric: str = "euclidean"):
    """Mixing metric

    As defined by Stuart et al., 2019.

    :param adata: AnnData object.
    :param batch_key: Key in adata.obs with batch information.
    :param embed: Key in adata_post.obsm with integrated embeddings.
    :param k: Rank of k-th nearest neighbor to consider.
    :param max_k: Maximum number of neighbors to search over.
    :param metric: Distance metric for nearest neighbor computation
    """
    batches = adata.obs[batch_key]
    unique_batches, batch_counts = np.unique(batches, return_counts=True)
    # Do not consider batches with fewer than k cells.
    unique_batches = unique_batches[batch_counts >= k]

    nn = NearestNeighbors(n_neighbors=max_k + 1, metric=metric).fit(adata.obsm[embed])
    _, indices = nn.kneighbors(adata.obsm[embed])
    indices = indices[:, 1:]
    all_ranks = []
    for ind in indices:
        batch = batches.iloc[ind]

        ranks = []
        for b in unique_batches:
            equal_batch = np.where(batch == b)[0]
            ranks.append(max_k if len(equal_batch) < k else equal_batch[k-1])
        all_ranks.append(np.median(ranks) / max_k)

    return np.mean(all_ranks)

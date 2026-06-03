import numpy as np
from sklearn.neighbors import NearestNeighbors


def adgbd(adata, batch_key, embed, n_neighbors: int = 50, metric: str = "euclidean"):
    """Average deviation from global batch distribution

    :param adata: AnnData object.
    :param batch_key: Key in adata.obs containing batch information.
    :param embed: Key in adata.obsm to use for nearest neighbors when n_neighbors
        is provided.
    :param n_neighbors: Number of nearest neighbors to use. Default: use the
        pre-computed nearest-neighbors in adata.
    :param metric: Distance metric for nearest neighbor computation
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric).fit(adata.obsm[embed])
    _, indices = nn.kneighbors(adata.obsm[embed])
    indices = indices[:, 1:]

    batch_counts = adata.obs[batch_key].value_counts(normalize=True)

    scores = []
    for i, indices in enumerate(indices):
        batch = adata.obs[batch_key].iloc[i]

        score = 1 - np.abs(
            (adata.obs[batch_key].iloc[indices] == batch).sum() / n_neighbors
            - batch_counts.loc[batch]
        )
        scores.append(score)
    return np.mean(scores)

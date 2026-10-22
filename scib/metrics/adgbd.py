import numpy as np
from sklearn.neighbors import NearestNeighbors


def adgbd(adata, batch_key, embed, n_neighbors: int = 50, metric: str = "euclidean") -> float:
    """Average deviation from global batch distribution (ADGBD)

    Computes the average deviation from global batch distribution
    proposed by Qian et al. 2022 "scINSIGHT for interpreting single-cell 
    gene expression from biologically heterogeneous data"

    doi:10.1186/s13059-022-02649-3

    :param adata: AnnData object.
    :param batch_key: Key in adata.obs containing batch information.
    :param embed: Key in adata.obsm to use for nearest neighbors when n_neighbors
        is provided.
    :param n_neighbors: Number of nearest neighbors to use.
    :param metric: Distance metric for nearest neighbor computation
    :return: The average deviation from global batch distribution metric.
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

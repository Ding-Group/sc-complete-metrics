import numpy as np
from sklearn.neighbors import NearestNeighbors


def over_correction(
    adata,
    label_key,
    embed,
    n_neighbors: int = 50,
    metric: str = "euclidean"
) -> float:
    """Over-correction score (OCS)

    Computes the over-correction score from Xiong et al. 2022
    "Online single-cell data integration through projecting heterogeneous 
    datasets into a common cell-embedding space"

    doi:10.1038/s41467-022-33758-z

    :param adata: AnnData object
    :param label_key: Key in adata.obs with cell type labels
    :param embed: Key in adata.obsm with integrated embedding.
    :param n_neighbors: Number of nearest neighbors. Default: 50.
    :param metric: Distance metric for nearest neighbor computation
    :return: Over-correction score
    """
    if adata.n_obs < 2:
        return float("nan")

    labels = adata.obs[label_key].to_numpy()

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric).fit(adata.obsm[embed])
    _, indices = nn.kneighbors(adata.obsm[embed])
    indices = indices[:, 1:]

    return over_correction_from_graph(labels, indices)


def over_correction_from_graph(labels, ind) -> float:
    """
    Compute over-correction score (STRICT reproduction of Eq. 18/19).
    Input:
      labels : (n,) array, cell-type labels (NO NaN)
      ind    : (n,k) array, neighbor index matrix (NO -1, NO self)
    Return:
      float in [0,1], higher means more over-correction
    """
    labels = np.asarray(labels)
    n, k = ind.shape

    if ind.shape[0] != n:
        raise ValueError("ind.shape[0] must equal number of labels")

    same = (labels[ind] == labels[:, None])  # Eq19
    purity = same.sum() / (n * k)

    # Turn into a positive score where higher value is better.
    # Not like the paper, not going to do 1 - score
    return purity

import typing

import numpy as np
import sklearn

# SOURCE:
# https://github.com/gao-lab/Cell_BLAST/blob/3a9cf2402a7dace62cfa2c4e32c77d6af4b77f2f/Cell_BLAST/metrics.py


_identity = lambda x, y: 1 if x == y else 0


def nnmap(
    adata,
    label_key,
    embed,
    k: float = 0.01,
    metric: str = "euclidean",
    similarity: typing.Callable = _identity,
    n_jobs: int = 1,
) -> float:
    """
    Nearest-neighbor mean average precision.
    """
    x = adata.obsm[embed]
    y = adata.obs[label_key].to_numpy()

    if k < 1:
        k = y.shape[0] * k
    k = np.round(k).astype(int)
    nearestNeighbors = sklearn.neighbors.NearestNeighbors(
        n_neighbors=min(y.shape[0], k + 1), metric=metric, n_jobs=n_jobs
    )
    nearestNeighbors.fit(x)
    nni = nearestNeighbors.kneighbors(x, return_distance=False)
    return mean_average_precision(y, y[nni[:, 1:]], similarity=similarity)


def _average_precision(r: np.ndarray) -> float:
    cummean = np.cumsum(r) / (np.arange(r.size) + 1)
    mask = r > 0
    if np.any(mask):
        return cummean[mask].mean()
    return 0.0


def mean_average_precision(
    true: np.ndarray,
    hits: np.ndarray,
    similarity: typing.Callable[[typing.Any, typing.Any], float] = _identity,
) -> float:
    r"""
    Mean average precision

    Parameters
    ----------
    true
        True label
    hits
        Hit labels
    similarity
        Function that defines the similarity between labels

    Returns
    -------
    map
        Mean average precision
    """
    r = np.apply_along_axis(np.vectorize(similarity), 0, hits, true)
    return np.apply_along_axis(_average_precision, 1, r).mean()
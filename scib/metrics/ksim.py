import sklearn

def ksim(
    adata,
    label_key,
    embed,
    beta: float = 0.9,
    k: int = 25,
    metric: str = "minkowski",
    n_jobs: int = 1
) -> float:
    """
    kSIM from Pegasus.
    """
    x = adata.obsm[embed]
    y = adata.obs[label_key].to_numpy()

    nearestNeighbors = sklearn.neighbors.NearestNeighbors(
        n_neighbors=min(y.shape[0], k + 1), metric=metric, n_jobs=n_jobs
    )
    nearestNeighbors.fit(x)
    nni = nearestNeighbors.kneighbors(x, return_distance=False)

    nni = nni[:, 1:]

    same = (y[nni] == y[:, None])
    correct_rates = same.sum(axis=1) / k

    kSIM_accept_rate = (correct_rates >= beta).sum() / adata.n_obs

    return kSIM_accept_rate

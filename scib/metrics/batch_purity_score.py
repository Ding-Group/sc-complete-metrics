# scib/metrics/batch_purity_score.py
import numpy as np
import pandas as pd
from ..utils import check_adata, check_batch


def batch_purity_score(
    adata,
    cluster_key: str = "cluster",
    batch_key: str = "batch",
    weighted: bool = True,
) -> float:
    """Batch purity score.

    Batch purity score: for each cluster, compute the fraction of cells that
    belong to the most frequent batch within that cluster. Then average these
    fractions across clusters. If `weighted` is True, use cluster sizes as
    weights; otherwise use the arithmetic mean.

    Range is [0, 1]. Smaller values indicate purer batches within clusters
    (worse mixing).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    cluster_key : str
        Column in `adata.obs` with cluster labels.
    batch_key : str
        Column in `adata.obs` with batch labels.
    weighted : bool
        If True, compute a size-weighted average across clusters.
        If False, compute an unweighted mean.

    Returns
    -------
    float
        The batch purity score. Returns `np.nan` for empty inputs.
    """
    check_adata(adata)
    check_batch(cluster_key, adata.obs)
    check_batch(batch_key, adata.obs)

    clusters = adata.obs[cluster_key]
    batches = adata.obs[batch_key]

    # Ensure categorical dtype
    if not pd.api.types.is_categorical_dtype(clusters):
        clusters = clusters.astype("category")
    if not pd.api.types.is_categorical_dtype(batches):
        batches = batches.astype("category")

    df = pd.DataFrame({"cluster": clusters, "batch": batches})
    if df.empty:
        return np.nan

    purities = []
    sizes = []
    for _, sub in df.groupby("cluster", observed=True):
        if sub.empty:
            continue
        # Proportions of batches within this cluster
        vc = sub["batch"].value_counts(normalize=True, dropna=False)
        purities.append(vc.iloc[0])  # maximum proportion
        sizes.append(len(sub))

    if not purities:
        return np.nan

    purities = np.array(purities, dtype=float)
    sizes = np.array(sizes, dtype=float)

    if weighted:
        w = sizes / sizes.sum()
        return 1 - float((purities * w).sum())
    else:
        return 1 - float(purities.mean())

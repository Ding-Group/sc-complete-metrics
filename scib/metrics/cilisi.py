import numpy as np
import scanpy as sc
import pandas as pd

from ..utils import check_adata, check_batch
from .lisi import ilisi_graph


def cilisi(
    adata,
    batch_key,
    label_key,
    embed=None,
    k0: int = 90,
    metric: str = "euclidean",
    subsample=None,
    scale: bool = True,
    n_cores: int = 1,
    verbose: bool = False,
    return_all: bool = False
):
    """Average iLISI per cell type.

    :param adata: AnnData object.
    :param batch_key: Key in adata.obs containing batch information.
    :param label_key: Key in adata.obs containing cell type information.
    :param embed: Key in adata.obsm containing the embedding.
    :param k0: Number of nearest neighbors for iLISI computation.
    :param metric: Distance metric for nearest neighbor computation.
    :param subsample: Percentage of observations used for LISI subsampling.
    :param scale: Whether to scale the score between 0 and 1.
    :param n_cores: Number of cores used for LISI computation.
    :param verbose: Whether to print progress messages.
    :param return_all: Whether to return scores for all cell types.
    :return: Mean CiLISI score, or scores for all cell types if return_all is True.
    """
    check_adata(adata)
    check_batch(batch_key, adata.obs)
    check_batch(label_key, adata.obs)

    values = []
    for label in np.unique(adata.obs[label_key]):
        subset = adata[adata.obs[label_key] == label].copy()
        if len(np.unique(subset.obs[batch_key])) == 1:
            print(f"Skipping {label}, only one batch")
            continue
        val = ilisi_graph(
            subset,
            batch_key,
            type_="embed",
            use_rep=embed,
            k0=k0,
            metric=metric,
            subsample=subsample,
            scale=scale,
            n_cores=n_cores,
            verbose=verbose,
        )
        values.append(val)

    if return_all:
        return pd.DataFrame(values, index=np.unique(adata.obs[label_key]), columns=['CILISI'])
    return np.mean(values)

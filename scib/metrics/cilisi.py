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
    subsample=None,
    scale=True,
    n_cores=1,
    verbose=False,
    return_all: bool = False
):
    """
    Average iLISI per cell type
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
            subsample=subsample,
            scale=scale,
            n_cores=n_cores,
            verbose=verbose,
        )
        values.append(val)

    if return_all:
        return pd.DataFrame(values, index=np.unique(adata.obs[label_key]), columns=['CILISI'])
    return np.mean(values)

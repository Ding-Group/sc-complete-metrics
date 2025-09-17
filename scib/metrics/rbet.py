import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from ..utils import check_adata, check_batch

# TODO: Redo this

def rbet(
    adata: AnnData,
    batch_key: str,
    label_key: str,
    rare_celltypes: list = None,
    rare_threshold: float = 0.01
) -> float:
    """
    Rare cell-type Batch Effect Test (RBET).

    Parameters
    ----------
    adata
        AnnData object with X as the gene expression matrix.
    batch_key
        Column name in adata.obs for batch labels.
    label_key
        Column name in adata.obs for cell type (or other biological label).
    rare_celltypes
        If provided, compute only for these types; otherwise automatically select based on rare_threshold.
    rare_threshold
        Threshold (fraction of total cells) to define rare cell types (default 1%).

    Returns
    -------
    A float: smaller values indicate weaker batch effect (better).
    """
    # Basic checks
    check_adata(adata)
    check_batch(batch_key, adata.obs)
    check_batch(label_key, adata.obs)

    obs = adata.obs
    total_cells = obs.shape[0]

    # 1) Automatically or manually select rare types
    if rare_celltypes is None:
        freq = obs[label_key].value_counts(normalize=True)
        rare_celltypes = freq[freq < rare_threshold].index.tolist()

    if len(rare_celltypes) == 0:
        raise ValueError(
            "No rare cell types found. "
            "Please lower rare_threshold or specify rare_celltypes manually."
        )

    # 2) Compute score for each rare type
    scores = []
    for ct in rare_celltypes:
        mask = (obs[label_key] == ct).values
        if mask.sum() < 2:
            # skip if too few samples
            continue

        # Subset expression matrix
        X_sub = adata.X[mask]
        if sparse.issparse(X_sub):
            X_sub = X_sub.toarray()

        df = pd.DataFrame(
            X_sub,
            columns=adata.var_names,
            index=np.where(mask)[0]
        )
        df['batch'] = obs.loc[mask, batch_key].values

        # Mean expression per batch
        mean_by_batch = df.groupby('batch').mean()
        # Overall variance of each gene within this cell type
        var_all = df.drop(columns='batch').var()

        # For each gene compute (max_mean - min_mean) / var
        diff = mean_by_batch.max() - mean_by_batch.min()
        diff_norm = diff / var_all.replace(0, np.nan)

        scores.append(diff_norm.mean())

    if not scores:
        raise RuntimeError(
            "All rare cell types were skipped due to insufficient samples; cannot compute RBET."
        )

    return float(np.nanmean(scores))

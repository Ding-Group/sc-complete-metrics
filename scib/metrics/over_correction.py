# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

def over_correction_from_graph(labels, ind) -> float:
    """
    Compute over-correction score (STRICT reproduction of Eq. 18/19).
    Input:
      labels : (n,) array, cell-type labels (NO NaN)
      ind    : (n,k) array, neighbor index matrix (NO -1, NO self)
    Return:
      float in [0,1], higher means more over-correction
    """
    # 1. Convert labels to numpy array
    labels = np.asarray(labels)
    n, k = ind.shape

    # 2. Check dimension consistency
    if ind.shape[0] != n:
        raise ValueError("ind.shape[0] must equal number of labels")

    same = (labels[ind] == labels[:, None])  # Eq19
    purity = same.sum() / (n * k)

    # Turn into a positive score where higher value is better.
    # Not like the paper, not going to do 1 - score
    return purity


def _topk_from_csr(csr: sparse.csr_matrix, k: int) -> np.ndarray:
    """
    From a CSR graph, return (n,k) top-k neighbor indices per row.
    - Remove self anywhere in each row
    - No padding/duplicates; require each row >= k after removing self
    - prefer_large=True for connectivities; False for distances
    """
    n = csr.shape[0]
    out = np.empty((n, k), dtype=int)
    for i in range(n):
        s, e = csr.indptr[i], csr.indptr[i+1]
        cols = csr.indices[s:e]
        # drop self anywhere
        m = (cols != i)
        cols = cols[m]
        out[i] = cols
    return out


def over_correction(
    adata,
    label_key,
    n_neighbors=None,
    embed=None
) -> float:
    """Over-correction

    :param adata: AnnData object
    :param label_key: Key in adata.obs with cell type labels
    :param n_neighbors: Number of nearest neighbors. If None, default to graph computed
        in adata.obsp['distances'].
    :param embed: Key in adata.obsm with integrated embedding, to be used if
        k provided.
    """
    if adata.n_obs < 2:
        return float("nan")

    labels = adata.obs[label_key].to_numpy()

    graph_key = 'distances'
    if n_neighbors is not None:
        if f'neighbors_{n_neighbors}' not in adata.uns.keys():
            sc.pp.neighbors(
                adata, use_rep=embed, n_neighbors=n_neighbors, key_added=f"neighbors_{n_neighbors}"
            )
        graph_key = f"neighbors_{n_neighbors}_distances"
    else:
        n_neighbors = adata.uns['neighbors']['params']['n_neighbors']

    D = adata.obsp[graph_key].tocsr()
    ind = _topk_from_csr(D, n_neighbors)

    # 4. Call core function
    return over_correction_from_graph(labels, ind)

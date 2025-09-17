# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
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
    n = labels.shape[0]

    # 2. Check dimension consistency
    if ind.shape[0] != n:
        raise ValueError("ind.shape[0] must equal number of labels")

    # **STRICT checks**
    if pd.isna(labels).any():
        raise ValueError("labels contain NaN; strict reproduction requires complete labels")
    if (ind < 0).any():
        raise ValueError("ind contains negative placeholders; strict reproduction requires exactly k neighbors per cell")

    # 3. Mask valid labels  
    valid = ~pd.isna(labels)  # all True under strict mode

    # 4. Cast labels to object type for safe comparison
    labels = labels.astype(object)

    # **5. Compare each cell to its k neighbors -> boolean matrix (n × k)
    same = (labels[ind] == labels[:, None])  # Eq19

    # 6. Only keep pairs where both cell and neighbor have valid labels
    valid_pairs = valid[:, None] & valid[ind]  

    # **7. Count total number of valid pairs
    total = int(valid_pairs.sum())
    # STRICT: total must be n * k
    if total == 0:
        return float("nan")

    # **8. Count number of equal-label pairs
    equal_cnt = int((same & valid_pairs).sum())  # Eq18

    # **9. Compute purity = equal / (n*k)  
    purity = equal_cnt / float(total)  # here total == n*k

    # **10. Final score = 1 - purity
    score = 1.0 - purity  # Eq18

    # 11. Clip result into [0,1]
    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0
    return float(score)

    


def _topk_from_csr(csr: sparse.csr_matrix, k: int, prefer_large: bool) -> np.ndarray:
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
        vals = csr.data[s:e]
        # drop self anywhere
        m = (cols != i)
        cols, vals = cols[m], vals[m]
        if cols.size < k:
            raise ValueError(f"Row {i} has only {cols.size} neighbors after removing self; strict reproduction needs >= k.")
        if prefer_large:
            top = np.argpartition(-vals, k-1)[:k]
            idx = top[np.argsort(-vals[top])] 
        else:
            top = np.argpartition(vals, k-1)[:k]
            idx = top[np.argsort(vals[top])]
        out[i] = cols[idx]
    return out


def over_correction(
    adata_pre,
    adata_post=None,
    *,
    label_key=None,
    k=None
) -> float:
    """
    Over-correction score using AnnData object (STRICT reproduction).
    Only uses existing nearest-neighbor graph and selects true top-k.
    Does NOT rebuild neighbors.
    """
    # 1. Choose which AnnData to use
    adata = adata_post if adata_post is not None else adata_pre
    n = int(adata.n_obs)
    if n < 2:
        return float("nan")

    # 2. Find label column
    if label_key is None:
        for cand in ("cell_type", "celltype", "CellType", "label", "labels"):
            if cand in adata.obs:
                label_key = cand
                break
    if label_key is None:
        raise ValueError("No label column found for strict reproduction")
    labels = adata.obs[label_key].to_numpy()
    if pd.isna(labels).any():
        raise ValueError("labels contain NaN; strict reproduction requires complete labels")

    # 3. Get neighbor indices (true top-k, self removed)
    if k is None:
        # try to infer from neighbors params; otherwise default to 30
        k = int(adata.uns.get("neighbors", {}).get("params", {}).get("n_neighbors", 30))
    k = int(k)
    if k <= 0 or k >= n:
        raise ValueError("k must be in [1, n-1] for strict reproduction")

    if "distances" in adata.obsp:
        D = adata.obsp["distances"].tocsr()
        ind = _topk_from_csr(D, k, prefer_large=False)
    elif "connectivities" in adata.obsp:
        C = adata.obsp["connectivities"].tocsr()
        ind = _topk_from_csr(C, k, prefer_large=True)
    elif "neighbors" in adata.uns and "indices" in adata.uns["neighbors"]:
        raw = adata.uns["neighbors"]["indices"]
        if raw.shape[1] < k:
            raise ValueError("neighbors.indices has fewer than k columns")
        ind = np.empty((n, k), dtype=int)
        for i in range(n):
            row = raw[i]
            row = row[row != i]
            if row.size < k:
                raise ValueError(f"Row {i} has only {row.size} neighbors after removing self")
            ind[i] = row[:k]
    else:
        raise ValueError("No nearest-neighbor graph found. Please run sc.pp.neighbors().")

    # 4. Call core function
    return over_correction_from_graph(labels, ind)

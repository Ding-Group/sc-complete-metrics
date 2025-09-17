# -*- coding: utf-8 -*-
import numpy as np
from itertools import combinations
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from universal_divergence import estimate
import time

cal_min = 30  # Minimum number of cells threshold

# Vectorized entropy calculation: from count matrix → entropy per row
def entropy_vec_from_counts(counts, base=None):
    counts = counts.astype(float)
    totals = counts.sum(axis=-1, keepdims=True)
    probs = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ent = -(probs * np.log(probs))
        ent = np.nan_to_num(ent, nan=0.0, posinf=0.0, neginf=0.0).sum(axis=-1)
    if base is not None:
        ent = ent / np.log(base)
    return ent

# Pre-compute kNN neighbors once (reuse later, drop self-neighbor)
def precompute_neighbors(X, k, algorithm='auto', metric='euclidean'):
    n = X.shape[0]
    k_eff = min(k, n-1)
    nbrs = NearestNeighbors(n_neighbors=k_eff+1, algorithm=algorithm, metric=metric)
    nbrs.fit(X)
    _, indices_all = nbrs.kneighbors(X, return_distance=True)
    return indices_all[:, 1:]  # drop self

# Compute entropy on a subset using cached neighbor matrix
def cal_entropy_fast(indices_all, idx_specific, dataset_labels_0based, base=None):
    sel = np.where(idx_specific)[0]
    if sel.size == 0:
        return []
    neigh = indices_all[sel]                 # (m, k)
    labs = dataset_labels_0based[neigh]      # (m, k)
    D = int(dataset_labels_0based.max()) + 1
    m, k = labs.shape
    counts = np.zeros((m, D), dtype=np.int32)
    rows = np.repeat(np.arange(m), k)
    counts[rows, labs.ravel()] += 1
    return entropy_vec_from_counts(counts, base=base).tolist()

# Optimized evaluate_scores: reuse kNN, vectorized entropy, faster silhouette
def evaluate_scores_fast(div_ent_code, sil_code, cell_labels, dataset_labels,
                         num_datasets, div_ent_dim, sil_dim, sil_dist, k_entropy=100):
    # PCA dimension reduction
    if div_ent_code.shape[1] > div_ent_dim:
        div_ent_code = PCA(n_components=div_ent_dim).fit_transform(div_ent_code)
    if sil_code.shape[1] > sil_dim:
        sil_code = PCA(n_components=sil_dim).fit_transform(sil_code)

    # Map dataset_labels → 0-based for vectorized bincount
    _, inv_d = np.unique(dataset_labels, return_inverse=True)
    d0 = inv_d

    # Precompute kNN neighbors once
    indices_all = precompute_neighbors(div_ent_code, k=k_entropy, algorithm='auto', metric='euclidean')

    div_pq, div_qp, ents = [], [], []
    for d1, d2 in combinations(range(1, num_datasets+1), 2):
        idx1 = (dataset_labels == d1)
        idx2 = (dataset_labels == d2)
        mutual = np.intersect1d(np.unique(cell_labels[idx1]), np.unique(cell_labels[idx2]))
        idx1_mut = idx1 & np.isin(cell_labels, mutual)
        idx2_mut = idx2 & np.isin(cell_labels, mutual)

        # Divergence (P||Q and Q||P)
        if idx1_mut.sum() >= cal_min and idx2_mut.sum() >= cal_min:
            X1 = div_ent_code[idx1_mut]
            X2 = div_ent_code[idx2_mut]
            d12 = max(estimate(X1, X2, cal_min), 0)
            d21 = max(estimate(X2, X1, cal_min), 0)
            div_pq.append(d12); div_qp.append(d21)

        # Entropy (dataset-specific cells)
        idx_specific = (idx1 | idx2) & (~np.isin(cell_labels, mutual))
        if idx_specific.any():
            ent_list = cal_entropy_fast(indices_all, idx_specific, d0, base=None)
            ents.append(np.mean(ent_list))

    if len(ents) == 0:
        ents = [-1]

    # Silhouette mean score
    sil = silhouette_score(sil_code, cell_labels, metric=sil_dist)

    div_score = float(np.mean([np.mean(div_pq), np.mean(div_qp)])) if (len(div_pq) and len(div_qp)) else np.nan
    ent_score = float(np.mean(ents))
    sil_score = float(sil)
    return div_score, ent_score, sil_score

# Quick test function: run once and print runtime + results
def quick_test(div_ent_code, sil_code, cell_labels, dataset_labels,
               num_datasets, div_ent_dim=2, sil_dim=20, sil_dist='euclidean', k_entropy=100):
    t0 = time.time()
    d, e, s = evaluate_scores_fast(div_ent_code, sil_code, cell_labels, dataset_labels,
                                   num_datasets, div_ent_dim, sil_dim, sil_dist, k_entropy=k_entropy)
    t1 = time.time()
    print(f"[Optimized] div={d:.6f}, ent={e:.6f}, sil={s:.6f}, time={t1-t0:.3f}s")

# ---------- Main test ----------
if __name__ == "__main__":
    # Example random data 
    n, d1, d2, Dsets = 5000, 10, 30, 3
    rng = np.random.default_rng(0)
    div_ent_code = rng.normal(size=(n, d1))
    sil_code = rng.normal(size=(n, d2))
    cell_labels = rng.integers(0, 8, size=n)
    dataset_labels = rng.integers(1, Dsets+1, size=n)

    quick_test(div_ent_code, sil_code, cell_labels, dataset_labels,
               num_datasets=Dsets, div_ent_dim=2, sil_dim=20, sil_dist='euclidean', k_entropy=100)

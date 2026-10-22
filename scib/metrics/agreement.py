import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

from ..utils import check_adata, check_batch


def agreement(
    adata_pre,
    adata_post,
    batch_key,
    label_key,
    embed,
    n_comps: int = 50,
    n_neighbors: int = 15,
    metric: str = "euclidean",
) -> float:
    """Agreement score (AGREE)

    Computes the agreement score from Welch et al. "Single-cell multi-omic
    integration compares and contrasts features of brain cell identity".

    doi:10.1016/j.cell.2019.05.006

    :param adata_pre: AnnData object without integration
    :param adata_post: AnnData object with integrated embedding
    :param batch_key: Column in adata_pre.obs with batch information
    :param label_key: Column in adata.obs with ground truth labels
    :param embed: Key in adata_post.obsm with integrated embedding
    :param n_comps: Number of components to use for PCA
    :param n_neighbors: Number of neighbors to consider
    :param metric: Distance metric for nearest neighbor computation
    :return: Agreement score
    """
    check_adata(adata_pre)
    check_adata(adata_post)
    check_batch(batch_key, adata_pre.obs)
    check_batch(batch_key, adata_post.obs)
    check_batch(label_key, adata_post.obs)

    per_cell_jaccard = []

    # Calculate low-dimensional representation of each batch in the raw data
    for c_key in adata_pre.obs[batch_key].unique():
        subset_pre = adata_pre[adata_pre.obs[batch_key] == c_key].copy()

        try:
            sc.pp.highly_variable_genes(
                subset_pre, flavor="seurat_v3", layer="counts", n_top_genes=2000
            )
            sc.tl.pca(subset_pre, n_comps=n_comps)
        except ValueError:
            # If there are too few cells to take highly variable genes,
            # skip that step and just do PCA.
            sc.tl.pca(subset_pre, n_comps=n_comps)

        neighbors_use = min(n_neighbors, subset_pre.n_obs - 1)

        nn = NearestNeighbors(n_neighbors=neighbors_use + 1, metric=metric).fit(
            subset_pre.obsm["X_pca"]
        )
        _, indices = nn.kneighbors(subset_pre.obsm["X_pca"])
        indices1 = indices[:, 1:]

        subset_post = adata_post[adata_post.obs[batch_key] == c_key].copy()
        subset_post = adata_post[subset_pre.obs.index]  # Make sure the cells are in the same order
        nn = NearestNeighbors(n_neighbors=neighbors_use + 1, metric=metric).fit(
            subset_post.obsm[embed]
        )
        _, indices = nn.kneighbors(subset_post.obsm[embed])
        indices2 = indices[:, 1:]

        jaccards = []
        for i in range(subset_pre.n_obs):
            s1, s2 = set(indices1[i]), set(indices2[i])
            jaccards.append(len(s1 & s2) / len(s1 | s2))

        per_cell_jaccard.extend(jaccards)

    return np.mean(per_cell_jaccard)

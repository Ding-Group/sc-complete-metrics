import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors


def retained_knn(
    adata_pre, adata_post, batch_key, embed_post, n_neighbors: int = 50
):
    """Retained nearest neighbors

    From Mandric et al., 2020.

    :param adata_pre: Pre-integration AnnData.
    :param adata_post: Post-integration AnnData.
    :param batch_key: Key in both AnnData .obs with batch information
    :param embed_post: Post-integration embedding key in adata_post.obsm.
    :param n_neighbors: Number of nearest neighbors to consider.
    """
    batches = np.unique(adata_pre.obs[batch_key])

    neighbors_pre = []
    obs_names = []
    for batch in batches:
        subset = adata_pre[adata_pre.obs[batch_key] == batch].copy()
        # Compute the nearest neighbors for the subset
        n_comps = min(30, subset.n_obs - 1)

        if subset.n_obs <= n_neighbors:
            sc.pp.highly_variable_genes(subset)
            sc.tl.pca(subset, n_comps=n_comps)
        # If not, then don't need to compute PCA, the nearest neighbors are just all the cells.
        neighbors_to_use = min(subset.n_obs - 1, n_neighbors)

        nn = NearestNeighbors(n_neighbors=neighbors_to_use + 1).fit(subset.obsm["X_pca"])
        _, indices = nn.kneighbors(subset.obsm["X_pca"])
        indices = indices[:, 1:]

        neighbors_pre.extend([subset.obs_names[i] for i in indices])
        obs_names.extend(subset.obs_names.to_list())

    adata_post = adata_post[obs_names]

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(adata_post.obsm[embed_post])
    _, indices = nn.kneighbors(adata_post.obsm[embed_post])
    indices = indices[:, 1:]

    neighbors_post = [adata_post.obs_names[i] for i in indices]

    scores = []
    for i_pre, i_post in zip(neighbors_pre, neighbors_post):
        i_post = i_post[:len(i_pre)] # In case i_pre comes from fewer neighbors
        scores.append(len(np.intersect1d(i_pre, i_post)) / n_neighbors)

    return np.mean(scores)

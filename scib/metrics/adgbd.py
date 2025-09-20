import scanpy as sc
import numpy as np


def adgbd(adata, batch_key, n_neighbors=None, embed=None):
    """Average deviation from global batch distribution

    :param adata: AnnData object.
    :param batch_key: Key in adata.obs containing batch information.
    :param n_neighbors: Number of nearest neighbors to use. Default: use the
        pre-computed nearest-neighbors in adata.
    :param embed: Key in adata.obsm to use for nearest neighbors when n_neighbors
        is provided.
    """
    # Add graph for pptpc
    if n_neighbors is not None:
        if f'neighbors_{n_neighbors}' not in adata.uns.keys():
            sc.pp.neighbors(
                adata, use_rep=embed, n_neighbors=n_neighbors, key_added=f"neighbors_{n_neighbors}"
            )
        graph_key = f"neighbors_{n_neighbors}_distances"
    else:
        n_neighbors = adata.uns["neighbors"]["params"]["n_neighbors"]
        graph_key = "distances"

    knn_graph = adata.obsp[graph_key]
    # Cell counts in order to adjust threshold if fewer than n_neighbors
    # cells in the data
    batch_counts = adata.obs[batch_key].value_counts(normalize=True)
    neighbors = [
        knn_graph.indices[knn_graph.indptr[i] : knn_graph.indptr[i + 1]]
        for i in range(knn_graph.shape[0])
    ]

    scores = []
    for i, indices in enumerate(neighbors):
        indices = indices[1:]  # exclude self

        batch = adata.obs[batch_key].iloc[i]

        score = 1 - np.abs(
            (adata.obs[batch_key].iloc[indices] == batch).sum() / n_neighbors
            - batch_counts.loc[batch]
        )
        scores.append(score)
    return np.mean(scores)

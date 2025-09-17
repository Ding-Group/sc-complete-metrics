import numpy as np
import scanpy as sc

from ..utils import check_adata, check_batch


def pptpc(adata, batch_key, label_key, embed=None, n_neighbors=100):
    """Proportion of Positive and True Positive Cells

    :param adata: AnnData object.
    :param batch_key: Key in adata.obs with batch labels.
    :param label_key: Key in adata.obs with cell type labels.
    :param embed: Key in adata.obsm with integrated embedding to compute
        nearest neighbor graph.
    :param n_neighbors: Number of nearest neighbors per cell. If None,
        the graph in adata.obsp['distances'] will be used.
    """
    check_adata(adata)
    check_batch(batch_key, adata.obs)
    check_batch(label_key, adata.obs)

    # Add graph for pptpc
    if n_neighbors is not None:
        sc.pp.neighbors(
            adata, use_rep=embed, n_neighbors=n_neighbors, key_added="pptpc"
        )
        graph_key = "pptpc_distances"
    else:
        n_neighbors = adata.uns["neighbors"]['params']["n_neighbors"]
        graph_key = "distances"

    # Categorize cells into "positive" and "negative" cells, where positive
    # cells are surrounded by at least 50% of its own cell type label.
    knn_graph = adata.obsp[graph_key]
    # Cell counts in order to adjust threshold if fewer than n_neighbors 
    # cells in the data
    cell_counts = adata.obs[label_key].value_counts()
    neighbors = [
        knn_graph.indices[knn_graph.indptr[i] : knn_graph.indptr[i + 1]]
        for i in range(knn_graph.shape[0])
    ]

    positive = []
    for i, indices in enumerate(neighbors):
        indices = indices[1:] # exclude self

        cell_type = adata.obs[label_key].iloc[i]
        threshold = min(n_neighbors, cell_counts.loc[cell_type]) / 2
        if (adata.obs[label_key].iloc[indices] == cell_type).sum() >= threshold:
            positive.append(i)

    # Of the positive cells, compute the true positive cells
    true_positive = []
    props = adata.obs.groupby(label_key)[batch_key].value_counts(normalize=True)

    for i in positive:
        indices = neighbors[i]

        neighborhood = adata.obs[batch_key].iloc[indices].value_counts()
        cell_type = adata.obs[label_key].iloc[i]

        kpi = props.loc[cell_type] * min(cell_counts.loc[cell_type], n_neighbors)
        lower = (kpi - 3 * np.sqrt(kpi * (1 - props.loc[cell_type]))).clip(lower=0).loc[neighborhood.index]
        upper = (kpi + 3 * np.sqrt(kpi * (1 - props.loc[cell_type]))).sort_index().loc[neighborhood.index]

        if (lower <= neighborhood).all() and (neighborhood <= upper).all():
            true_positive.append(i)
    return len(true_positive) / adata.n_obs

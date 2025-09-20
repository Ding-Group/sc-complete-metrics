import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..utils import check_adata, check_batch


def pptpc(adata, batch_key, label_key, embed, n_neighbors=100):
    """Proportion of Positive and True Positive Cells

    :param adata: AnnData object.
    :param batch_key: Key in adata.obs with batch labels.
    :param label_key: Key in adata.obs with cell type labels.
    :param embed: Key in adata.obsm with integrated embedding.
    :param n_neighbors: Number of nearest neighbors per cell.
    """
    check_adata(adata)
    check_batch(batch_key, adata.obs)
    check_batch(label_key, adata.obs)

    # Add graph for pptpc
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(adata.obsm[embed])
    _, indices = nn.kneighbors(adata.obsm[embed])
    indices = indices[:, 1:]

    cell_counts = adata.obs[label_key].value_counts()

    positive = []
    for i, ind in enumerate(indices):
        cell_type = adata.obs[label_key].iloc[i]
        threshold = min(n_neighbors, cell_counts.loc[cell_type]) / 2
        if (adata.obs[label_key].iloc[ind] == cell_type).sum() >= threshold:
            positive.append(i)

    # Of the positive cells, compute the true positive cells
    true_positive = []
    props = adata.obs.groupby(label_key)[batch_key].value_counts(normalize=True)

    for i in positive:
        ind = indices[i]

        neighborhood = adata.obs[batch_key].iloc[ind].value_counts()
        cell_type = adata.obs[label_key].iloc[i]

        kpi = props.loc[cell_type] * min(cell_counts.loc[cell_type], n_neighbors)
        lower = (kpi - 3 * np.sqrt(kpi * (1 - props.loc[cell_type]))).clip(lower=0).loc[neighborhood.index]
        upper = (kpi + 3 * np.sqrt(kpi * (1 - props.loc[cell_type]))).sort_index().loc[neighborhood.index]

        if (lower <= neighborhood).all() and (neighborhood <= upper).all():
            true_positive.append(i)
    return len(true_positive) / adata.n_obs

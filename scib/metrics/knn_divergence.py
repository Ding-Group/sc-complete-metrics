import numpy as np
from universal_divergence import estimate


# https://github.com/txWang/BERMUDA/blob/1e1e031b7dc59bca98a494729f151905bf05e13b/evaluate.py#L39
def knn_divergence(adata, label_key, batch_key, embed, min_cells=30):
    """KNN Divergence

    As proposed by Wang et al., 2019.

    :param adata: AnnData object.
    :param label_key: Key in adata.obs containing cell type information.
    :param batch_key: Key in adata.obs containing batch information.
    :param embed: Key in adata.obsm with integrated embedding.
    :param min_cells: Minimum number of cells for estimation.
    """
    div_ent_code = adata.obsm[embed]
    cell_labels = np.asarray(adata.obs[label_key])
    batches = np.asarray(adata.obs[batch_key])
    unique_batches = np.unique(batches)

    div_pq = []
    div_qp = []
    for i in range(len(unique_batches)):
        for j in range(i + 1, len(unique_batches)):
            idx1 = batches == unique_batches[i]
            idx2 = batches == unique_batches[j]

            labels = np.intersect1d(
                np.unique(cell_labels[idx1]), np.unique(cell_labels[idx2])
            )
            idx1_mutual = np.logical_and(idx1, np.isin(cell_labels, labels))
            idx2_mutual = np.logical_and(idx2, np.isin(cell_labels, labels))

            if np.sum(idx1_mutual) >= min_cells and np.sum(idx2_mutual) >= min_cells:
                div_pq.append(
                    max(
                        estimate(
                            div_ent_code[idx1_mutual, :],
                            div_ent_code[idx2_mutual, :],
                            min_cells,
                        ),
                        0,
                    )
                )
                div_qp.append(
                    max(
                        estimate(
                            div_ent_code[idx2_mutual, :],
                            div_ent_code[idx1_mutual, :],
                            min_cells,
                        ),
                        0,
                    )
                )

    div_score = (sum(div_pq) / len(div_pq) + sum(div_qp) / len(div_qp)) / 2
    return div_score

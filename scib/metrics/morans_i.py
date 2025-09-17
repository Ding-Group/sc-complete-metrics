import numpy as np
import pandas as pd
import scanpy as sc
from scanpy.metrics._morans_i import _morans_i
from scipy.spatial import distance_matrix

from ..utils import check_adata, check_batch


# Based on https://github.com/theislab/cross_system_integration/blob/e2958d3f18aa2971d1cb9dd7e191e998d0969678/notebooks/eval/cleaned/analysis/moransi_examples.ipynb
# and https://pmc.ncbi.nlm.nih.gov/articles/PMC10635119/#S9
def morans_i(
    adata_pre,
    adata_post,
    label_key,
    batch_key,
    embed,
    min_cells=500,
    moran_cutoff=0.2,
    max_genes=150,
    use_gene_score=False,
):
    """Moran's I

    Proposed by Hrovatin et al. Higher values indicate that sub-cell type
    biological variations are conserved.

    :param adata_pre: AnnData object prior to integration.
    :param adata_post AnnData object with integrated embedding and nearest
        neighbor graph.
    :param label_key: Key in adata.obs for cell labels.
    :param batch_key: Key in adata.obs for desired batch labels.
    :param embed: Key in adata.obsm with integrated embeddings.
    :param min_cells: Minimum cells for a label x batch pair to be considered.
    :param moran_cutoff: Minimum Moran's I cutoff for gene to be included.
    :param max_genes: Maximum number of genes per label x batch pair.
    :param use_gene_score: Whether to use gene scores instead of average across genes.

    :return: Moran's I
    """
    check_adata(adata_pre)
    check_batch(label_key, adata_pre.obs)
    check_batch(batch_key, adata_pre.obs)

    selected_genes = sub_variable_genes(
        adata_pre, label_key, batch_key, min_cells, moran_cutoff
    )
    if not selected_genes:
        raise ValueError(
            "No genes were found to be suitable for Moran's I."
            "Please decrease min_cells and/or moran_cutoff."
        )

    if max_genes <= 0:
        raise ValueError("Maximum number of genes must be positive.")

    pre_post_ratios = []  # Ratio of post/pre Moran's I
    if use_gene_score:
        for gene_set in selected_genes:
            # Compute the gene set scores
            score_name = f"{gene_set[label_key]}_{gene_set[batch_key]}_score"
            genes = gene_set["genes"].sort_values(ascending=False)
            genes = list(genes.index[:max_genes])
            sc.tl.score_genes(
                adata_pre,
                gene_list=genes,
                score_name=score_name,
                use_raw=False,
            )

            # Calculate the Moran's I on the unintegrated data
            adata_pre_sub = adata_pre[
                (adata_pre.obs[label_key] == gene_set[label_key])
                & (adata_pre.obs[batch_key] == gene_set[batch_key])
            ]
            # Use the PCA embedding from the unintegrated data
            sc.pp.neighbors(adata_pre_sub, use_rep="X_pca")
            morans_i_pre = _morans_i(
                adata_pre_sub.obsp["connectivities"],
                vals=adata_pre_sub.obs[[score_name]].T,
            )

            adata_post_sub = adata_post[
                (adata_post.obs[label_key] == gene_set[label_key])
                & (adata_post.obs[batch_key] == gene_set[batch_key])
            ]
            # Use adata_post.obsm[embed] for integrated data
            sc.pp.neighbors(adata_post_sub, use_rep=embed)
            morans_i_post = _morans_i(
                adata_post_sub.obsp["connectivities"],
                vals=adata_post_sub.obs[[score_name]].T
            )
            pre_post_ratios.append(morans_i_post / morans_i_pre)
    else:
        for gene_set in selected_genes:
            genes = gene_set["genes"].index[:max_genes]

            adata_pre_sub = adata_pre[
                (adata_pre.obs[label_key] == gene_set[label_key])
                & (adata_pre.obs[batch_key] == gene_set[batch_key])
            ]
            # Use the PCA embedding from the unintegrated data
            sc.pp.neighbors(adata_pre_sub, use_rep="X_pca")
            morans_i_pre = _morans_i(
                adata_pre_sub.obsp["connectivities"],
                vals=adata_pre_sub[:, genes].X.T
            )

            adata_post_sub = adata_post[
                (adata_post.obs[label_key] == gene_set[label_key])
                & (adata_post.obs[batch_key] == gene_set[batch_key])
            ]
            # Use adata_post.obsm[embed] for integrated data
            sc.pp.neighbors(adata_post_sub, use_rep=embed)
            morans_i_post = _morans_i(
                adata_post_sub.obsp["connectivities"],
                vals=adata_post_sub[:, genes].X.T
            )
            
            pre_post_ratios.append(morans_i_post / morans_i_pre)

    if use_gene_score:
        return np.mean(pre_post_ratios)
    else:
        return np.concatenate(pre_post_ratios).mean()

# Based on https://github.com/theislab/cross_system_integration/blob/e2958d3f18aa2971d1cb9dd7e191e998d0969678/notebooks/data/adipose_sc_sn_updated.py
def sub_variable_genes(adata, label_key, batch_key, min_cells, moran_cutoff):
    """Compute genes used for Moran's I

    Genes that have a high Moran's I in a particular label x batch pair
    are returned. Only label x batch pairs with a sufficient number of
    cells are considered.

    :param adata: AnnData object prior to integration.
    :param label_key: Key in adata.obs for cell labels.
    :param batch_key: Key in adata.obs for desired batch labels.
    :param min_cells: Minimum number of cells for the label x batch pair to be considered.
    :param moran_cutoff: Minimum Moran's I for the gene to be included.
    """
    # Find label x batch pairs with >= min_cells
    groups = adata.obs.groupby([label_key, batch_key]).size()
    groups = groups[groups >= min_cells]

    if len(groups) == 0:
        return []

    data = []
    for (label, batch), group in groups.items():
        adata_sub = adata[
            (adata.obs[label_key] == label).values
            & (adata.obs[batch_key] == batch).values
        ].copy()
        # Remove lowly expr genes before Moran's I computation as they will be less likely relevant
        # As this is done per small cell group within sample+cell type and HVGs there is not many genes (200-500)
        # so all can be used for Moran's I computation
        sc.pp.filter_genes(adata_sub, min_cells=adata_sub.shape[0] * 0.1)
        # Compute embedding of group
        sc.pp.pca(adata_sub, n_comps=15)
        sc.pp.neighbors(adata_sub, n_pcs=15)
        # Compute I
        morans_i = _morans_i(adata_sub.obsp["connectivities"], vals=adata_sub.X.T)
        # Save data
        morans_i = pd.DataFrame({"morans_i": morans_i}, index=adata_sub.var_names)
        morans_i[label_key] = label
        morans_i[batch_key] = batch
        data.append(morans_i)
    data = pd.concat(data, axis=0)

    selected = list()
    for group, data_sub in data.groupby([label_key, batch_key]):
        group = dict(zip([label_key, batch_key], group))
        group["genes"] = (data_sub.query("morans_i>=@moran_cutoff")["morans_i"] + 1) / 2
        selected.append(group)

    return selected

# REMOVED

# # scib/scib/metrics/coefficient_variation.py

# import numpy as np
# import pandas as pd
# from ..utils import check_adata

# def coefficient_of_variation(
#     adata_pre,
#     adata_post=None,
#     gene_list=None,
#     **kwargs
# ):
#     """
#     Coefficient of variation (CV) across batch embedding means.

#     Can be called as:
#       coefficient_variation(adata, gene_list=adata.var_names)
#     or, for Runner compatibility:
#       coefficient_variation(adata, adata_int, gene_list=adata.var_names)

#     :param adata_pre: AnnData before integration (only embedding used here)
#     :param adata_post: AnnData after integration (ignored)
#     :param gene_list: Iterable of gene names or, in our case, list of embedding column names
#     :return: float CV of batch means
#     """
#     check_adata(adata_pre)

#     # If gene_list provided but embedding lives in adata.obsm['X_pca'], ignore gene_list
#     # We assume embedding is X_pca
#     if "X_pca" not in adata_pre.obsm:
#         raise ValueError("Embedding 'X_pca' not found in adata.obsm")
#     X = adata_pre.obsm["X_pca"]

#     # batch labels
#     if "batch" not in adata_pre.obs:
#         raise KeyError("batch not found in adata.obs")
#     batch_labels = adata_pre.obs["batch"].to_numpy()

#     # Build a DataFrame of shape (n_cells, n_dims+1)
#     df = pd.DataFrame(X)
#     df["batch"] = batch_labels

#     # Compute per-batch mean embedding
#     batch_means = df.groupby("batch").mean().values  # (n_batches, n_dims)
#     mean = np.mean(batch_means)
#     std  = np.std(batch_means)
#     if mean == 0:
#         return 0.0
#     return float(std / mean)

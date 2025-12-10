import numpy as np


def ldf_diff(
    adata_pre,
    adata_post,
    batch_key: str,
    embed_pre: str,
    embed_post: str,
    n_dim=None,
    k: int = 75,
    return_per_cell = False
):
    if n_dim is None:
        n_dim = min(adata_post.obsm[embed_post].shape[1], adata_pre.obsm[embed_pre].shape[1])

    if (
        n_dim > adata_post.obsm[embed_post].shape[1]
        or n_dim > adata_pre.obsm[embed_pre].shape[1]
    ):
        raise ValueError(
            f"{n_dim} should not be greater than the number of dimensions in both"
            "adata_pre.obsm[embed_pre] and adata_post.obsm[embed_post]"
        )

    X_embed_pre = np.asarray(adata_pre.obsm[embed_pre], dtype=float)
    X_embed_pre = X_embed_pre[:, :n_dim]
    X_embed_post = np.asarray(adata_post.obsm[embed_post], dtype=float)
    X_embed_post = X_embed_post[:, :n_dim]

    # Batch labels and cell names (R will factor + droplevels).
    batch = adata_post.obs[batch_key].astype("category").astype(str).to_numpy()
    cell_names = adata_post.obs_names.to_list()

    k = min(k, adata_post.obs[batch_key].value_counts().min() - 1)

    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        from rpy2.robjects import numpy2ri, StrVector
        from rpy2.robjects import conversion, default_converter
        from rpy2.robjects import numpy2ri
    except Exception as e:
        raise RuntimeError(
            "rpy2 is required but not available. Install with `pip install rpy2`."
        ) from e

    numpy_converter = default_converter + numpy2ri.converter

    with conversion.localconverter(numpy_converter):
        importr("SingleCellExperiment")
        importr("SummarizedExperiment")
        importr("CellMixS")
        r = ro.r

        # 3.Build SCE (minimal when embedding is provided).
        n_cells = adata_post.n_obs
        r.assign("n_cells", int(n_cells))

        r(
            "sce <- SingleCellExperiment::SingleCellExperiment("
            "assays = list(logcounts = matrix(0, nrow=1, ncol=n_cells)))"
        )

        # colnames and group column (factor + droplevels in R).
        r.assign("cell_names", StrVector(cell_names))
        r("colnames(sce) <- cell_names")
        r.assign("group_vals", StrVector(batch))
        r.assign("group_key", batch_key)
        r(
            "SummarizedExperiment::colData(sce)[, group_key] <- base::factor(group_vals)"
        )
        r(
            "SummarizedExperiment::colData(sce) <- "
            "S4Vectors::DataFrame(SummarizedExperiment::colData(sce))"
        )
        r(
            "SummarizedExperiment::colData(sce)[, group_key] <- "
            "base::droplevels(SummarizedExperiment::colData(sce)[, group_key])"
        )

        # reducedDim
        r.assign("embed_mat", np.asarray(X_embed_pre, dtype=float))
        r.assign("dim_name", "X_embed_pre")
        r("SingleCellExperiment::reducedDim(sce, dim_name) <- embed_mat")

        r.assign("embed_mat", np.asarray(X_embed_post, dtype=float))
        r.assign("dim_name", "X_embed_post")
        r("SingleCellExperiment::reducedDim(sce, dim_name) <- embed_mat")

        r.assign("k", int(k))
        r.assign("n_dim", int(n_dim))

        r("sce_pre_list <- list()")
        r(f"""
for (s in unique(colData(sce)${batch_key})) {{
sce_pre_list[[s]] <- sce[, colData(sce)${batch_key} == s]
}}
            """)
        r(
            f"sce_ldf <- ldfDiff(sce_pre_list, sce, k = {k}, group = '{batch_key}', dim_red = 'X_embed_pre', dim_combined = 'X_embed_post', n_dim = {n_dim})"
        )
        cms_vals = np.array(
            r('as.numeric(SummarizedExperiment::colData(sce_ldf)[, "diff_ldf"])'),
            dtype=float,
        )


    if return_per_cell:
        return cms_vals
    return float(np.nanmean(cms_vals))

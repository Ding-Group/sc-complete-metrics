import numpy as np

from ..utils import check_adata


def cell_specific_mixing_score(
    adata,
    batch_key,
    embed,
    k: int = 70,
    n_dim=None,
    cell_min=10,
    smooth=True,
    unbalanced=False,
    res_name=None,
    return_per_cell=False,
) -> float:
    """Cell-specific mixing score

    Computes the cell-specific mixing score from Lutge et al., 2022 "CellMixS: quantifying
    and visualizing batch effects in single-cell RNA-seq data", a wrapper around ``CellMixS::cms`` via rpy2.
    For each cell, the distances to its k nearest neighbors in the embedding are grouped by batch and the
    groupwise distance distributions are compared with the K-sample Anderson-Darling test.
    The score is the p-value of this test, so higher values indicate better batch mixing.

    doi:10.26508/lsa.202001004

    :param adata: AnnData object containing the data and annotations.
    :param batch_key: Key in adata.obs for batch labels.
    :param embed: Key in adata.obsm for the embedding used to compute nearest neighbors.
    :param k: Number of nearest neighbors.
    :param n_dim: Number of embedding dimensions to use. Default: all dimensions.
    :param cell_min: Minimum number of cells from a batch in the neighborhood for the batch to be included in
        the test.
    :param smooth: Whether to smooth the score within each neighborhood.
    :param unbalanced: If True, neighborhoods containing a single batch are set to NA.
    :param res_name: Suffix for the result column in the R object (``cms`` or ``cms.<res_name>``).
    :param return_per_cell: If True, return the per-cell scores instead of their mean.
    :return: Mean score across cells, ignoring NaN values, or an array of per-cell scores if
        ``return_per_cell`` is True.
    """
    check_adata(adata)

    if n_dim is None:
        n_dim = adata.obsm[embed].shape[1]
    elif n_dim > adata.obsm[embed].shape[1]:
        raise ValueError("n_dim is greater than embedding dim")

    try:
        import anndata2ri
        import rpy2.robjects as ro
        from rpy2.robjects import conversion, default_converter, numpy2ri
        from rpy2.robjects.packages import importr
    except Exception as e:
        raise RuntimeError("Failure importing rpy2") from e

    numpy_converter = default_converter + numpy2ri.converter + anndata2ri.converter
    with conversion.localconverter(numpy_converter):
        try:
            importr("SingleCellExperiment")
            importr("SummarizedExperiment")
            importr("BiocParallel")
            importr("CellMixS")
            r = ro.r

            r.assign("k", int(k))
            r.assign("dim_red", embed)
            r.assign("group_key", batch_key)
            r.assign("n_dim", int(n_dim))
            r.assign("cell_min", int(cell_min))
            r.assign("smooth", bool(smooth))
            r.assign("unbalanced", bool(unbalanced))
            r.assign("res_name", ro.NULL if res_name is None else str(res_name))
            r("bp <- BiocParallel::SerialParam()")

            del adata.uns
            r.assign("sce", adata)

            r(
                "sce <- CellMixS::cms(sce, k = k, group = group_key, "
                "dim_red = dim_red, n_dim = n_dim, cell_min = cell_min, "
                "smooth = smooth, unbalanced = unbalanced, res_name = res_name, BPPARAM = bp)"
            )

            result_col = "cms" if res_name is None else f"cms.{res_name}"
            r.assign("result_col", result_col)
            cms_vals = np.array(
                r("as.numeric(SummarizedExperiment::colData(sce)[, result_col])"),
                dtype=float,
            )

        except Exception as e:
            raise RuntimeError(
                "Calling R CellMixS::cms failed. Ensure R and required R packages are installed."
            ) from e

    if return_per_cell:
        return cms_vals
    return float(np.nanmean(cms_vals))

from ..utils import check_adata
import numpy as np
import scanpy as sc
import scipy.sparse as sp


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
):
    """Cell-specific mixing score (CMS) wrapper around CellMixS::cms via rpy2.

    Computes kNN in a subspace (default PCA) and compares groupwise distance
    distributions using the K-sample Anderson-Darling test.

    Behavior
    --------
    - If a reduced dimension named ``dim_red_name`` exists, use it.
    - Else if ``r_run_pca_when_embed_missing=True``, pass expression to R and let
      ``scater::runPCA`` compute PCA on the ``logcounts`` assay.
    - Otherwise compute PCA in Python (Scanpy) and pass it as
      ``reducedDim(sce, dim_red_name)``.

    Parameters
    ----------
    adata : AnnData
        Cells x genes matrix and metadata.
    batch_key : str
        Column in ``adata.obs`` with batch/group labels.
    k : int
        Number of nearest neighbors.
    embed : str, optional
        Key in ``adata.obsm`` for the embedding to use (default ``"X_pca"``).
    n_dim : int, optional
        Number of embedding dimensions to use (default all).
    cell_min : int, optional
        Minimum cells from each group for the AD test (default 10).
    smooth : bool, optional
        Smooth CMS within each neighborhood (default True).
    unbalanced : bool, optional
        If True, single-batch neighborhoods are set to NA (default False).
    res_name : str or None, optional
        Suffix for result column (``cms`` or ``cms.<res_name>``).
    return_per_cell : bool, optional
        If True return the per-cell vector, else return the NaN-mean.
    """
    check_adata(adata)

    if n_dim is None:
        n_dim = adata.obsm[embed].shape[1]
    elif n_dim > adata.obsm[embed].shape[1]:
        raise ValueError("n_dim is greater than embedding dim")

    try:
        import anndata2ri
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        from rpy2.robjects import numpy2ri
        from rpy2.robjects import conversion, default_converter
    except Exception as e:
        raise RuntimeError(
            "Failure importing rpy2"
        ) from e

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
            r.assign('group_key', batch_key)
            r.assign("n_dim", int(n_dim))
            r.assign("cell_min", int(cell_min))
            r.assign("smooth", bool(smooth))
            r.assign("unbalanced", bool(unbalanced))
            r.assign("res_name", ro.NULL if res_name is None else str(res_name))
            r('bp <- BiocParallel::SerialParam()')

            del adata.uns
            r.assign("sce", adata)

            r('sce <- CellMixS::cms(sce, k = k, group = group_key, '
            'dim_red = dim_red, n_dim = n_dim, cell_min = cell_min, '
            'smooth = smooth, unbalanced = unbalanced, res_name = res_name, BPPARAM = bp)')

            result_col = "cms" if res_name is None else f"cms.{res_name}"
            r.assign("result_col", result_col)
            cms_vals = np.array(
                r('as.numeric(SummarizedExperiment::colData(sce)[, result_col])'),
                dtype=float,
            )

        except Exception as e:
            raise RuntimeError(
                "Calling R CellMixS::cms failed. Ensure R and required R packages are installed."
            ) from e

    if return_per_cell:
        return cms_vals
    return float(np.nanmean(cms_vals))

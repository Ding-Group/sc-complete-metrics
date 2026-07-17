import numpy as np
import scanpy as sc
import warnings
warnings.simplefilter("ignore", FutureWarning)

# FutureWarning
warnings.simplefilter("ignore", FutureWarning)

try:
    from scanpy._utils import renamed_arg
except ImportError:
    from .._package_tools import renamed_arg

from ..utils import split_batches


def precompute_hvg_batch(adata, batch, features, n_hvg=500, save_hvg=False):
    """Compute highly variable genes per batch.

    :param adata: AnnData object.
    :param batch: Key in adata.obs containing batch information.
    :param features: Features used when splitting batches.
    :param n_hvg: Number of highly variable genes to compute per batch.
    :param save_hvg: Whether to save highly variable genes in adata.uns.
    :return: Dictionary of highly variable genes per batch if save_hvg is False.
    """
    adata_list = split_batches(adata, batch, hvg=features)
    hvg_dir = {}
    for i in adata_list:
        sc.pp.filter_genes(i, min_cells=1)
        n_hvg_tmp = np.minimum(n_hvg, int(0.5 * i.n_vars))
        if n_hvg_tmp < n_hvg:
            print(i.obs[batch].iloc[0] + " has less than the specified number of genes")
            print("Number of genes: " + str(i.n_vars))
        hvg = sc.pp.highly_variable_genes(
            i, flavor="cell_ranger", n_top_genes=n_hvg_tmp, inplace=False
        )
        hvg_dir[i.obs[batch].iloc[0]] = i.var.index[hvg["highly_variable"]]

    if save_hvg:
        adata.uns["hvg_before"] = hvg_dir
    else:
        return hvg_dir


@renamed_arg("batch", "batch_key")
def hvg_overlap(adata_pre, adata_post, batch_key, n_hvg=500, verbose=False):
    """Highly variable gene overlap metric.

    :param adata_pre: AnnData object before integration.
    :param adata_post: AnnData object after integration.
    :param batch_key: Key in adata.obs containing batch information.
    :param n_hvg: Number of highly variable genes to compute per batch.
    :param verbose: Whether to print progress messages.
    :return: Mean highly variable gene overlap across batches.
    """
   
    hvg_post = adata_post.var_names

    
    adata_post_list = split_batches(adata_post, batch_key)
    overlap = []

    
    hvg_pre_list = precompute_hvg_batch(
        adata_pre, batch_key, hvg_post, n_hvg=n_hvg
    )

    for ad_post in adata_post_list:
        sc.pp.filter_genes(ad_post, min_cells=1)
        batch_var = ad_post.obs[batch_key].iloc[0]
        n_hvg_tmp = len(hvg_pre_list[batch_var])

        if verbose:
            print(n_hvg_tmp)

        tmp_pre = hvg_pre_list[batch_var]
        hvg_post = sc.pp.highly_variable_genes(
            ad_post, flavor="cell_ranger", n_top_genes=n_hvg_tmp, inplace=False
        )
        tmp_post = ad_post.var.index[hvg_post["highly_variable"]]
        n_hvg_real = np.minimum(len(tmp_pre), len(tmp_post))
        overlap.append(
            len(set(tmp_pre).intersection(set(tmp_post))) / n_hvg_real
        )
    return np.mean(overlap)

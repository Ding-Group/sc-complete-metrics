import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.special
from sklearn.metrics.cluster import adjusted_rand_score

try:
    from scanpy._utils import renamed_arg
except ImportError:
    from .._package_tools import renamed_arg

from ..utils import check_adata, check_batch


@renamed_arg("group1", "cluster_key")
@renamed_arg("group2", "label_key")
def ari(adata, label_key, cluster_key, implementation=None) -> float:
    """Adjusted Rand Index (ARI)

    The adjusted rand index is a chance-adjusted rand index, which evaluates the pair-wise accuracy of clustering vs.
    ground truth label assignments.
    The score ranges between 0 and 1 with larger values indicating better conservation of the data-driven cell identity
    discovery after integration compared to annotated labels.

    :param adata: anndata object with cluster assignments in ``adata.obs[cluster_key]``
    :param cluster_key: string of column in adata.obs containing cluster assignments
    :param label_key: string of column in adata.obs containing labels
    :param implementation: if set to 'sklearn', uses sklearn's implementation,
        otherwise native implementation is taken
    :return: ARI score

    This function can be applied to all integration output types.
    The ``adata`` must contain cluster assignments that are based off the knn graph given or derived from the integration
    method output.
    For this metric you need to include all steps that are needed for clustering.
    See :ref:`preprocessing` for more information on preprocessing.

    **Examples**

    .. code-block:: python

        # feature output
        scib.pp.reduce_data(
            adata, n_top_genes=2000, batch_key="batch", pca=True, neighbors=True
        )
        scib.me.cluster_optimal_resolution(adata, cluster_key="cluster", label_key="celltype")
        scib.me.ari(adata, cluster_key="cluster", label_key="celltype")

        # embedding output
        sc.pp.neighbors(adata, use_rep="X_emb")
        scib.me.cluster_optimal_resolution(adata, cluster_key="cluster", label_key="celltype")
        scib.me.ari(adata, cluster_key="cluster", label_key="celltype")

        # knn output
        scib.me.cluster_optimal_resolution(adata, cluster_key="cluster", label_key="celltype")
        scib.me.ari(adata, cluster_key="cluster", label_key="celltype")

    """

    check_adata(adata)
    check_batch(cluster_key, adata.obs)
    check_batch(label_key, adata.obs)

    cluster_key = adata.obs[cluster_key].to_numpy()
    label_key = adata.obs[label_key].to_numpy()

    if len(cluster_key) != len(label_key):
        raise ValueError(
            f"different lengths in cluster_key ({len(cluster_key)}) and label_key ({len(label_key)})"
        )

    if implementation == "sklearn":
        return adjusted_rand_score(cluster_key, label_key)

    def binom_sum(x, k=2):
        return scipy.special.binom(x, k).sum()

    n = len(cluster_key)
    contingency = pd.crosstab(cluster_key, label_key)

    ai_sum = binom_sum(contingency.sum(axis=0))
    bi_sum = binom_sum(contingency.sum(axis=1))

    index = binom_sum(np.ravel(contingency))
    expected_index = ai_sum * bi_sum / binom_sum(n, 2)
    max_index = 0.5 * (ai_sum + bi_sum)

    return (index - expected_index) / (max_index - expected_index)


def ari_fixed(
    adata,
    label_key,
    cluster_key,
    cluster_function=None,
    embed=None,
    resolution: float = 2.0,
    **kwargs,
) -> float:
    """ARI fixed (ARIF)

    Version of ARI where clustering is applied with a very high resolution.
    Clusters are then mapped onto labels based on the most prevalent
    ground-truth label in the cluster.

    :param adata: AnnData object.
    :param label_key: Key in adata.obs containing ground-truth cell labels.
    :param cluster_key: name of column to be added to adata.obs during clustering.
        If the key already exists, it will be overwritten.
    :param cluster_function: a clustering function that takes an anndata.Anndata object.
        Default: Leiden clustering
    :param embed: key of embedding to use only if ``adata.uns['neighbors']`` is not
        defined, otherwise will be ignored
    :param resolution: Clustering resolution to use. Default: 2.0
    :param kwargs: arguments to pass to clustering function
    :return: ARI fixed score
    """
    if cluster_function is None:
        cluster_function = sc.tl.leiden

    if cluster_key is None:
        cluster_key = cluster_function.__name__

    if cluster_key in adata.obs.columns:
        warnings.warn(
            f"Overwriting existing key {cluster_key} in adata.obs", stacklevel=2
        )

    # check or recompute neighbours
    knn_rep = adata.uns.get("neighbors", {}).get("params", {}).get("use_rep")
    if embed is not None and embed != knn_rep:
        print(f"Recompute neighbors on rep {embed} instead of {knn_rep}")
        sc.pp.neighbors(adata, use_rep=embed)

    # call clustering function
    print(f"Cluster with {cluster_function.__name__}")
    cluster_function(adata, resolution=resolution, key_added=cluster_key, **kwargs)

    mapping = adata.obs.groupby(cluster_key)[label_key].agg(lambda x: x.mode().iloc[0])
    fixed_labels = adata.obs[cluster_key].map(mapping)
    return adjusted_rand_score(adata.obs[label_key], fixed_labels)

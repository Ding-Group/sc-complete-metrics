import warnings

from sklearn.metrics.cluster import normalized_mutual_info_score
import scanpy as sc

def nmi_fixed(
    adata,
    label_key,
    cluster_key,
    cluster_function=None,
    embed=None,
    resolution=2.0,
    **kwargs,
):
    """NMI fixed

    Version of NMI where clustering is applied with a very high resolution.
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
    return normalized_mutual_info_score(adata.obs[label_key], fixed_labels)

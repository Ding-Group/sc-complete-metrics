import igraph as ig

from ..utils import check_adata


def modularity(adata, label_key, cluster_key) -> float:
    """Modularity

    Compute the modularity of a clustering assignment.

    :param adata: AnnData object with a neighbors graph and cluster labels
    :param cluster_key: Column in adata.obs containing cluster labels
    :param label_key: (unused) included for signature consistency
    :return: Modularity score
    """
    check_adata(adata)

    if "connectivities" not in adata.obsp:
        raise ValueError(
            "Please run sc.pp.neighbors(...) first, so that the connectivity matrix is stored in adata.obsp['connectivities']"
        )
    conn = adata.obsp["connectivities"]

    sources, targets = conn.nonzero()
    g = ig.Graph(n=conn.shape[0], edges=list(zip(sources, targets)), directed=False)

    clusters = adata.obs[cluster_key].astype("category").cat.codes.values
    return g.modularity(clusters)

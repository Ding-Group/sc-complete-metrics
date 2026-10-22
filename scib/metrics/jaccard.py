import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score

from ..utils import check_adata, check_batch

# Source: https://github.com/theislab/cross_system_integration/tree/e2958d3f18aa2971d1cb9dd7e191e998d0969678

def jaccard_index(adata, label_key, cluster_key) -> float:
    """Jaccard index (JACC)

    Computes the micro-averaged Jaccard index between clustering assignments and true labels.
    Each cluster is first mapped to the most frequent true label among its cells (majority voting), and the
    Jaccard index ``TP / (TP + FP + FN)`` is then computed between the true labels and the mapped labels.

    :param adata: AnnData object.
    :param label_key: Key in ``adata.obs`` for groundtruth cell type labels.
    :param cluster_key: Key in ``adata.obs`` for cluster assignments.
    :return: Micro-averaged Jaccard index.
    """
    check_adata(adata)
    check_batch(cluster_key, adata.obs)
    check_batch(label_key, adata.obs)

    labels = adata.obs[label_key].to_numpy()
    clusters = adata.obs[cluster_key].to_numpy()

    labels_df = pd.DataFrame({"labels": labels, "clusters": clusters})
    cluster_map = (
        labels_df.groupby("clusters")["labels"]
        .agg(lambda x: pd.Series.mode(x)[0])
        .to_dict()
    )
    labels_df['labels_pred']=labels_df['clusters'].map(cluster_map)

    return jaccard_score(labels_df['labels'], labels_df['labels_pred'], average='micro')


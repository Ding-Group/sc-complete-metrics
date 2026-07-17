import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score

from ..utils import check_adata, check_batch

# Source: https://github.com/theislab/cross_system_integration/tree/e2958d3f18aa2971d1cb9dd7e191e998d0969678

def jaccard_index(adata, label_key, cluster_key):
    """
    Compute the Jaccard Index (micro-averaged) between predicted clusters and ground truth labels.

    Parameters:
    - adata: AnnData object
    - cluster_key: str, column in adata.obs with predicted cluster labels
    - label_key: str, column in adata.obs with ground truth labels

    Returns:
    - Jaccard Index (float)
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


def jaccard_kmeans(adata, label_key, embed, key_added='kmeans'):
    """Version of Jaccard where number of clusters fixed by KMeans"""
    kmeans = KMeans(n_clusters=adata.obs[label_key].nunique(), random_state=0).fit(adata.obsm[embed])
    adata.obs[key_added] = kmeans

    return jaccard_score(adata.obs[label_key], adata.obs[key_added], average='micro')

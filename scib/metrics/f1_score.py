import pandas as pd
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans

from ..utils import check_adata, check_batch


def f1_score_macro(adata, label_key, cluster_key):
    """
    Compute the macro-averaged F1 score between predicted clusters and ground truth labels.

    Parameters:
    - adata: AnnData object
    - cluster_key: str, column in adata.obs with predicted cluster labels
    - label_key: str, column in adata.obs with ground truth labels

    Returns:
    - Macro F1 score (float)
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

    return f1_score(labels_df['labels'], labels_df['labels_pred'], average='macro')


def f1_score_macro_kmeans(adata, label_key, embed, key_added='kmeans'):
    """Compute macro F1 score using KMeans clusters.

    :param adata: AnnData object.
    :param label_key: Key in adata.obs containing cell type information.
    :param embed: Key in adata.obsm containing the embedding.
    :param key_added: Key added to adata.obs for KMeans cluster assignments.
    :return: Macro F1 score.
    """
    kmeans = KMeans(n_clusters=adata.obs[label_key].nunique(), random_state=0).fit(adata.obsm[embed])
    adata.obs[key_added] = kmeans

    return f1_score(adata.obs[label_key], adata.obs[key_added], average='macro')

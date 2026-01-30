import pandas as pd
from sklearn.metrics import jaccard_score

from ..utils import check_adata, check_batch


# Source: https://github.com/theislab/cross_system_integration/tree/e2958d3f18aa2971d1cb9dd7e191e998d0969678

def jaccard_index(adata, label_key, cluster_key):
    """
    Compute the Jaccard Index (macro-averaged) between predicted clusters and ground truth labels.

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

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from ..utils import check_adata, check_batch


def clustering_accuracy(adata, label_key, cluster_key):
    """
    Compute clustering accuracy by matching predicted clusters to true labels.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing cluster and label annotations in adata.obs.
    label_key : str
        Key in adata.obs for ground truth labels.
    cluster_key : str
        Key in adata.obs for predicted cluster labels.

    Returns
    -------
    float
        Clustering accuracy after optimal one-to-one label matching.
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

    return accuracy_score(labels_df['labels'], labels_df['labels_pred'])

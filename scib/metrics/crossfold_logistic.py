import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from ..utils import check_adata, check_batch


def crossfold_logistic(
    adata,
    label_key,
    embed,
    n_splits=5,
    max_iter=1000,
    random_state=0,
):
    """
    Cross-fold logistic regression predicting true labels from cluster assignments.

    Can be called as:
      crossfold_logistic(adata, cluster_key='cluster', label_key='celltype')
    or (second AnnData ignored):
      crossfold_logistic(adata, adata_int, cluster_key='cluster', label_key='celltype')

    :param adata_pre: AnnData with .obs containing cluster_key and label_key
    :param label_key: Column in adata.obs to use as targets (true labels)
    :param embed: Key in adata.obsm with integrated embedding
    :param n_splits: Number of StratifiedKFold splits
    :param random_state: RNG seed for splitting
    :return: Mean classification accuracy across folds
    """
    # Basic checks
    check_adata(adata)
    check_batch(label_key, adata.obs)

    # Prepare data: cluster IDs as single‐column feature
    X = adata.obsm[embed]
    y = adata.obs[label_key].to_numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    acc_scores = []

    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=max_iter)
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        acc_scores.append(accuracy_score(y[test_idx], y_pred))

    return float(np.mean(acc_scores))

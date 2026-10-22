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
) -> float:
    """Cross-batch logistic regression (FLOGREG)

    Measures the ability of a logistic regression classifier to predict cell type labels of cells
    from a held-out fold from the remaining folds.

    :param adata: AnnData object
    :param label_key: Column in ``adata.obs`` with groundtruth cell type labels
    :param embed: Key in ``adata.obsm`` with integrated embedding
    :param n_splits: Number of StratifiedKFold splits
    :param max_iter: Maximum number of iterations during fitting logistic regression
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

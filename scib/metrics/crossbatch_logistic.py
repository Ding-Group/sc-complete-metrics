import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from ..utils import check_adata, check_batch


def crossbatch_logistic(
    adata,
    batch_key: str,
    label_key: str,
    embed: str,
    max_iter: int = 1000,
) -> float:
    """Cross-batch logistic regression

    Measures the ability of a logistic regression classifier to predict cell type labels of cells from one batch
    using cells from all other batches. For each batch, a classifier is trained on the cells from all other
    batches and evaluated on the held-out batch. If the integration does not remove batch effects, the held-out
    cells may be out-of-distribution relative to the training data and the accuracy will be low.

    :param adata: AnnData object containing the data and annotations.
    :param batch_key: Key in adata.obs for batch labels.
    :param label_key: Key in adata.obs for true cell type labels.
    :param embed: Key in adata.obsm for the integrated embedding.
    :param max_iter: Maximum number of iterations for the logistic regression solver.
    :return: Mean classification accuracy across held-out batches.
    """
    check_adata(adata)
    check_batch(batch_key, adata.obs)
    check_batch(label_key, adata.obs)

    X = adata.obsm[embed]
    y = adata.obs[label_key].to_numpy()
    batch = adata.obs[batch_key].astype("category").cat.codes.to_numpy()

    unique_batches = np.unique(batch)
    scores = []

    for i in range(len(unique_batches)):
        train_idx = np.where(batch != unique_batches[i])[0]
        test_idx = np.where(batch == unique_batches[i])[0]

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        clf = LogisticRegression(max_iter=max_iter)
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], y_pred)
        scores.append(acc)

    return float(np.mean(scores)) if scores else np.nan

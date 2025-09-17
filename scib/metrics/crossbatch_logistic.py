from typing import Literal
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
    """
    Train a logistic regression classifier on one batch and evaluate on another batch.
    Average the score across all ordered train→test batch pairs that have at least two
    shared classes.

    Parameters
    ----------
    adata : AnnData
        AnnData with embeddings in .obsm[use_rep] or in .X
    label_key : str
        Column in .obs with biological labels (e.g., cell types)
    batch_key : str
        Column in .obs with batch assignments
    embed: str
        Key in .obsm with integrated embedding
    max_iter : int
        max_iter for LogisticRegression

    Returns
    -------
    float
        Mean score across valid train→test pairs, or np.nan if none are valid.
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

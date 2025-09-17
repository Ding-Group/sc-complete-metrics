from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

from ..utils import check_adata, check_batch


def crossbatch_knn(adata, batch_key, label_key, embed, n_neighbors=21):
    """
    Cross-batch k-NN classifier:
    Train classifier on all but one batch, test on the held out one.
    Returns average accuracy over all batch pairs.
    """
    check_adata(adata)
    check_batch(batch_key, adata.obs)

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

        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], y_pred)
        scores.append(acc)

    return float(np.mean(scores)) if scores else np.nan

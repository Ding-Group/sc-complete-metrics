import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from ..utils import check_adata, check_batch


def crossbatch_knn(
    adata, batch_key, label_key, embed, n_neighbors: int = 21, metric: str = "euclidean"
) -> float:
    """Cross-batch KNN classifier

    Train classifier on all but one batch, test on the held out one.
    Returns average accuracy over all batch pairs.

    :param adata: AnnData object
    :param batch_key: Column in ``adata.obs`` with batch information
    :param label_key: Column in ``adata.obs`` with cell type information
    :param embed: Key in ``adata.obsm`` with the integrated embedding
    :param n_neighbors: Number of nearest neighbors to use
    :param metric: Distance metric for nearest neighbor computation
    :return: Mean cross-batch KNN label classification accuracy
    """
    check_adata(adata)
    check_batch(batch_key, adata.obs)

    X = adata.obsm[embed]
    y = adata.obs[label_key].astype("category").cat.codes.to_numpy()
    batch = adata.obs[batch_key].astype("category").cat.codes.to_numpy()

    unique_batches = np.unique(batch)
    scores = []

    for i in range(len(unique_batches)):
        train_idx = np.where(batch != unique_batches[i])[0]
        test_idx = np.where(batch == unique_batches[i])[0]

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], y_pred)
        scores.append(acc)

    return float(np.mean(scores)) if scores else np.nan

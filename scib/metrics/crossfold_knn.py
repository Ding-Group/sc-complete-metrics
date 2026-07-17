from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

from ..utils import check_adata, check_batch


def crossfold_knn(adata, label_key, embed, n_splits: int = 5, n_neighbors: int = 21, metric: str = "euclidean", random_state=0):
    """Cross-fold k-nearest neighbor classification.
    
    :param adata: AnnData object
    :param label_key: Column in ``adata.obs`` with cell type information
    :param embed: Key in ``adata.obsm`` with the integrated embedding
    :param n_splits: Number of folds
    :param n_neighbors: Number of nearest neighbors to use
    :param metric: Distance metric for nearest neighbor computation
    """
    check_adata(adata)
    check_batch(label_key, adata.obs)

    X = adata.obsm[embed]
    y = adata.obs[label_key].astype('category').cat.codes.to_numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    acc_scores = []

    for train_idx, test_idx in skf.split(X, y):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        acc_scores.append(accuracy_score(y[test_idx], y_pred))

    return np.mean(acc_scores)

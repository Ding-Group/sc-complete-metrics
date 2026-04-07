from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

from ..utils import check_adata, check_batch


def crossfold_knn(adata, label_key, embed, n_splits: int = 5, n_neighbors: int = 21, random_state=0):
    check_adata(adata)
    check_batch(label_key, adata.obs)

    X = adata.obsm[embed]
    y = adata.obs[label_key].to_numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    acc_scores = []

    for train_idx, test_idx in skf.split(X, y):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        acc_scores.append(accuracy_score(y[test_idx], y_pred))

    return np.mean(acc_scores)

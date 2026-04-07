import numpy as np


def adjusted_wallace_score(adata, label_key, cluster_key, direction="mean"):
    """Adjusted Wallace Score

    :param adata: AnnData object.
    :param label_key: Key in adata.obs containing cell type information.
    :param cluster_key: Key in adata.obs containing cluster information.
    """

    labels_true = adata.obs[label_key].astype("category").cat.codes
    labels_pred = adata.obs[cluster_key].astype("category").cat.codes

    _, y1c = np.unique(labels_pred, return_inverse=True)
    _, y2c = np.unique(labels_true, return_inverse=True)

    K1 = y1c.max() + 1
    K2 = y2c.max() + 1
    # contingency table
    cont = np.zeros((K1, K2))
    np.add.at(cont, (y1c, y2c), 1)

    n = int(cont.sum())
    if n < 2:
        return np.nan

    def comb2(x):
        return x * (x - 1) // 2

    a = comb2(cont).sum()
    a1 = comb2(cont.sum(1)).sum()
    a2 = comb2(cont.sum(0)).sum()
    b = a1 - a
    c = a2 - a

    # Wallace Score
    w1 = a / float(a + b) if (a + b) > 0 else np.nan  # 1->2
    w2 = a / float(a + c) if (a + c) > 0 else np.nan  # 2->1

    f1 = cont.sum(1)  # size y1
    f2 = cont.sum(0)  # size y2
    D1 = (f1 * (f1 - 1)).sum() / (n * (n - 1))  # for y1
    D2 = (f2 * (f2 - 1)).sum() / (n * (n - 1))  # for y2

    aw1 = (w1 - D2) / (1.0 - D2) if np.isfinite(w1) and (1.0 - D2) > 0 else np.nan
    aw2 = (w2 - D1) / (1.0 - D1) if np.isfinite(w2) and (1.0 - D1) > 0 else np.nan

    if direction == "1->2":  # pred_to_true
        return aw1
    if direction == "2->1":  # true_to_pred
        return aw2
    if direction == "mean":
        return (aw1 + aw2) / 2
    if direction == "both":
        return aw1, aw2
    raise ValueError("direction must be '1->2', '2->1', 'mean', or 'both'")

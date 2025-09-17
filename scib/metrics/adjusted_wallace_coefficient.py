import numpy as np

def adjusted_wallace_score(y1, y2, direction="1->2"):
    # transit to numpy
    y1 = np.asarray(y1).astype(str)
    y2 = np.asarray(y2).astype(str)
    if y1.shape[0] != y2.shape[0]:
        raise ValueError("y1 and y2 must have the same length")

    _, y1c = np.unique(y1, return_inverse=True)
    _, y2c = np.unique(y2, return_inverse=True)

    K1 = y1c.max() + 1
    K2 = y2c.max() + 1
    # contingency table
    cont = np.zeros((K1, K2), dtype=np.int64)
    np.add.at(cont, (y1c, y2c), 1)

    n = int(cont.sum())
    if n < 2:
        return np.nan

    def comb2(x): 
        x = x.astype(np.int64)
        return x * (x - 1) // 2

    a  = comb2(cont).sum()
    a1 = comb2(cont.sum(1)).sum()
    a2 = comb2(cont.sum(0)).sum()
    b  = a1 - a
    c  = a2 - a

    # Wallace Score
    w1 = a / float(a + b) if (a + b) > 0 else np.nan  # 1->2
    w2 = a / float(a + c) if (a + c) > 0 else np.nan  # 2->1

    
    f1 = cont.sum(1).astype(np.int64) # size y1
    f2 = cont.sum(0).astype(np.int64) # size y2
    D1 = (f1 * (f1 - 1)).sum() / (n * (n - 1))  # for y1
    D2 = (f2 * (f2 - 1)).sum() / (n * (n - 1))  # for y2

    aw1 = (w1 - D2) / (1.0 - D2) if np.isfinite(w1) and (1.0 - D2) > 0 else np.nan
    aw2 = (w2 - D1) / (1.0 - D1) if np.isfinite(w2) and (1.0 - D1) > 0 else np.nan

    if direction in ("1->2", "pred_to_true"):
        return aw1
    if direction in ("2->1", "true_to_pred"):
        return aw2
    if direction in ("both", None):
        return aw1, aw2
    raise ValueError("direction must be '1->2', '2->1', or 'both'")

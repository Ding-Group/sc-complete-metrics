# Source: https://github.com/codicef/BPS

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from ..utils import check_adata, check_batch


def batch_probing_score(
    adata,
    batch_key,
    embed,
    default_preprocess=False,
    n_jobs=-1,
    n_estimators=100,
    max_depth=10,
    max_iter=100,
):
    """Batch Probing Score

    Computes BPS-RF and BPS-LR metrics for the given dataset.
    Modification: since we use embeddings rather than gene expression for BPS,
    default_preprocess is default False.

    Parameters:
    - adata: AnnData.
    - batches: Labels (batch information).
    - embed: Key in adata.obsm with integrated embedding.

    Returns:
    - BPS_RF: BPS-RF score.
    - BPS_LR: BPS-LR score.
    """
    check_adata(adata)
    check_batch(batch_key, adata.obs)

    X = adata.obsm[embed]
    batches = adata.obs[batch_key].to_numpy()

    if default_preprocess:
        print("Preprocessing data...")
        X = StandardScaler().fit_transform(X)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced_subsample",
        n_jobs=n_jobs,
        random_state=42,
    )
    lr = LogisticRegression(
        max_iter=max_iter, C=1.0, n_jobs=n_jobs, class_weight="balanced"
    )

    # Compute BPS-RF (Random Forest)
    print("Computing BPS-RF...")
    rf_scores = cross_val_score(
        rf, X, batches, cv=5, scoring="roc_auc_ovr"
    )  # Calcoliamo ROC AUC
    BPS_RF = (np.mean(rf_scores) - 0.5) / 0.5

    if default_preprocess:
        X = safe_truncate(X, lower_percentile=0.1, upper_percentile=99)
    # Compute BPS-LR (Logistic Regression)
    print("Computing BPS-LR...")
    lr_scores = cross_val_score(lr, X, batches, cv=5, scoring="roc_auc_ovr")
    BPS_LR = (np.mean(lr_scores) - 0.5) / 0.5

    # Lower prediction accuracy is better
    return 1 - BPS_RF, 1 - BPS_LR


def safe_truncate(X, lower_percentile=1, upper_percentile=99):
    '''
    Truncates the values in X to the specified percentiles.
    '''
    lower_bound = np.percentile(X, lower_percentile, axis=0)
    upper_bound = np.percentile(X, upper_percentile, axis=0)
    X_clipped = np.clip(X, lower_bound, upper_bound)
    return X_clipped
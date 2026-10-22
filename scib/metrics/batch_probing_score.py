import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from ..utils import check_adata, check_batch

# Source: https://github.com/codicef/BPS

def per_label_batch_probing_score(
    adata,
    label_key,
    batch_key,
    embed,
    weighted: bool = True,
    min_cells: int = 100,
    default_preprocess: bool = False,
    n_jobs=-1,
    n_estimators=100,
    max_depth=10,
    max_iter=100
):
    """Batch probing score per cell type

    Computes the batch probing score for each cell type that has
    at least `min_cells`. Batch probing scores are averaged across
    cell types.
    
    :param adata: AnnData object.
    :param label_key: Key in adata.obs containing cell type information.
    :param batch_key: Key in adata.obs containing batch information.
    :param embed: Key in adata.obsm containing the embedding.
    :param weighted: Whether to return a weighted average across cell types or uniform average.
    :param min_cells: Number of minimum cells in a cell type to compute batch probing score.
    :param default_preprocess: Whether to standardize and truncate the embedding before scoring.
    :param n_jobs: Number of parallel jobs for model fitting.
    :param n_estimators: Number of trees for the random forest classifier.
    :param max_depth: Maximum depth of the random forest classifier.
    :param max_iter: Maximum number of iterations for logistic regression.
    :return: BPS-RF and BPS-LR scores.
    """
    rfs, lrs = [], []
    weights = []
    for label in np.unique(adata.obs[label_key]):
        adata_subset = adata[adata.obs[label_key] == label].copy()

        if len(np.unique(adata_subset.obs[batch_key])) < 2:
            print(f"Skipping {label} as it has <2 batches")
            continue
        if adata_subset.n_obs < min_cells:
            print(f"Skipping {label} as it has <{min_cells} cells")
            continue

        rf, lr = batch_probing_score(
            adata_subset,
            batch_key,
            embed,
            default_preprocess=default_preprocess,
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_iter=max_iter
        )
        if np.isnan(rf) or np.isnan(lr):
            continue
        rfs.append(rf)
        lrs.append(lr)

        if weighted:
            weights.append(adata_subset.n_obs / adata.n_obs)
        else:
            weights.append(1)

    return np.average(rfs, weights=weights), np.average(lrs, weights=weights)


# SOURCE: https://github.com/codicef/BPS/blob/main/src/bps.py
def batch_probing_score(
    adata,
    batch_key,
    embed,
    default_preprocess=False,
    n_jobs=-1,
    n_estimators=100,
    max_depth=10,
    max_iter=100,
) -> float:
    """Batch probing score.

    Computes BPS-RF and BPS-LR metrics for the given embedding from
    Codice et al. 2025 "Probing as a new technique to assess single-cell 
    RNA-seq batch correction"

    doi:10.1101/2025.05.12.653389

    :param adata: AnnData object.
    :param batch_key: Key in adata.obs containing batch information.
    :param embed: Key in adata.obsm containing the embedding.
    :param default_preprocess: Whether to standardize and truncate the embedding before scoring.
    :param n_jobs: Number of parallel jobs for model fitting.
    :param n_estimators: Number of trees for the random forest classifier.
    :param max_depth: Maximum depth of the random forest classifier.
    :param max_iter: Maximum number of iterations for logistic regression.
    :return: BPS-RF and BPS-LR scores.
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
    """Truncate values to the specified percentiles.

    :param X: Input matrix.
    :param lower_percentile: Lower percentile used for clipping.
    :param upper_percentile: Upper percentile used for clipping.
    :return: Clipped matrix.
    """
    lower_bound = np.percentile(X, lower_percentile, axis=0)
    upper_bound = np.percentile(X, upper_percentile, axis=0)
    X_clipped = np.clip(X, lower_bound, upper_bound)
    return X_clipped
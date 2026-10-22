import numpy as np

from ..utils import check_adata, check_batch


def chunked_corr_mse(X1, X2, chunk_size=5000) -> float:
    """Compute MSE between row-wise correlation matrices in chunks.

    :param X1: First input matrix.
    :param X2: Second input matrix.
    :param chunk_size: Number of rows processed in each chunk.
    :return: Mean squared error between row-wise correlation matrices.
    """
    def standardize(X):
        return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, ddof=1, keepdims=True)

    Z1, Z2 = standardize(X1), standardize(X2)
    n_features = Z1.shape[0]
    total_sq_diff = 0.0

    for i in range(0, n_features, chunk_size):
        end_i = min(i + chunk_size, n_features)

        row_chunk1 = np.dot(Z1[i:end_i], Z1.T) / (Z1.shape[1] - 1)
        row_chunk2 = np.dot(Z2[i:end_i], Z2.T) / (Z2.shape[1] - 1)
        
        total_sq_diff += np.sum((row_chunk1 - row_chunk2)**2)

    return total_sq_diff / (n_features**2)


def corr_mse(adata_pre, adata_post, batch_key, embed_post, embed_pre='X_pca') -> float:
    """Corr-MSE (CORRMSE)

    From Yi et al., 2025 "Benchmarking deep learning methods for
    biologically conserved single-cell integration"

    doi:10.1186/s13059-025-03869-z

    :param adata_pre: Pre-integration AnnData.
    :param adata_post: Post-integration AnnData.
    :param batch_key: Key in both AnnData .obs with batch information
    :param embed_post: Post-integration embedding key in adata_post.obsm.
    :param embed_pre: Pre-integration embedding key in adata_pre.obsm.
    :return: Corr-MSE
    """
    check_adata(adata_post)
    check_batch(batch_key, adata_pre.obs)
    check_batch(batch_key, adata_post.obs)

    batches = np.unique(adata_post.obs[batch_key])
    n_batches = len(batches)

    loss = 0
    for b in batches:
        adata_pre_sub = adata_pre[adata_pre.obs[batch_key] == b]
        adata_post_sub = adata_post[adata_post.obs[batch_key] == b]

        # Reorder just in case they aren't in the same order
        adata_post_sub = adata_post_sub[adata_pre_sub.obs_names]

        X1 = adata_pre_sub.obsm[embed_pre]
        X2 = adata_post_sub.obsm[embed_post]

        # corr1 = np.corrcoef(X1)
        # corr2 = np.corrcoef(X2)

        # loss += mse_loss(corr1, corr2)
        loss += chunked_corr_mse(X1, X2)
    return 1 - loss / n_batches

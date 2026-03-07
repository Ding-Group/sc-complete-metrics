import numpy as np

from ..utils import check_adata, check_batch


def scaled_variance(adata, batch_key, embed, n_comps=None):
    """Scaled variance

    :param adata: AnnData object.
    :param batch_key: Key in adata.obs with batch labels.
    :param embed: Key in adata.obsm with integrated embedding.
    :param n_comps: Number of dimensions to evaluate scaled variance.
        If not provided, scaled variance is evaluated across all dimensions.
    """
    check_adata(adata)
    check_batch(batch_key, adata.obs)

    if n_comps is None:
        # We can't really recommend a default value for n_comps because
        # in most methods, the embedding dims are not in order of 
        # decreasing importance.
        n_comps = adata.obsm[embed].shape[1]

    batches = np.unique(adata.obs[batch_key])

    svs = []
    for i in range(n_comps):
        component = adata.obsm[embed][:, i]
        denom = np.var(component)
        if denom == 0:
            continue

        for batch in batches:
            component_batch = adata.obsm[embed][adata.obs[batch_key] == batch, i]
            if len(component_batch) <= 1:
                continue

            # Closer the value is to 1, the better.
            svs.append(1 - np.abs(1 - np.var(component_batch) / denom))

    return np.mean(svs)

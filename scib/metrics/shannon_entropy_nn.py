import numpy as np
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numba import njit

from ..utils import check_adata, check_batch


def shannon_entropy_nn(adata, batch_key, label_key, embed, n_neighbors: int = 50):
    """Shannon entropy

    Compute the batch and cell type Shannon entropy within each cell's
    local neighbourhood.

    :param adata: AnnData object
    :param batch_key: Key in adata.obs with batch labels
    :param label_key: Key in adata.obs with cell type labels
    :param embed: Key in adata.obsm with integrated embedding,
    :param n_neighbors: Number of nearest neighbors. Default: 50
    :return: Shannon entropy
    """
    # Basic checks
    check_adata(adata)
    check_batch(batch_key, adata.obs)
    check_batch(label_key, adata.obs)

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(adata.obsm[embed])
    _, indices = nn.kneighbors(adata.obsm[embed])
    indices = indices[:, 1:]

    batches = adata.obs[batch_key].astype("category").cat.codes
    labels = adata.obs[label_key].astype("category").cat.codes
    n_batches, n_labels = len(np.unique(batches)), len(np.unique(labels))

    batch_entropy = []
    celltype_entropy = []

    for ind in indices:
        selected_batches = batches[ind]
        selected_labels = labels[ind]

        batch_entropy.append(entropy_(selected_batches, n_batches))
        celltype_entropy.append(entropy_(selected_labels, n_labels))

    # The lower the celltype entropy, the better
    return np.mean(batch_entropy), 1 - np.mean(celltype_entropy)


def entropy_(selected_labels, n_labels):
    """
    Shannon entropy calculation normalized by log(n_labels).

    :param labels: Labels for the group.
    :param n_labels: Number of unique batches.
    """
    # Compute frequency distribution
    _, counts = np.unique(selected_labels, return_counts=True)
    freqs = counts / counts.sum()

    # Shannon entropy (base = natural log, matches R log())
    h = entropy(freqs, base=2)

    # Normalize by log(n_labels)
    return h / np.log2(n_labels)


@njit
def Hbeta(D_row, beta):
    """
    Helper function for simpson index computation
    """
    D_row[np.isnan(D_row)] = 0
    P = np.exp(-D_row * beta)
    sumP = np.sum(P)
    if sumP == 0:
        H = 0
        P = np.zeros(len(D_row))
    else:
        H = np.log(sumP) + beta * np.sum(D_row * P) / sumP
        P /= sumP

    return H, P


def get_weighted_probs(D_act, target_perplexity, tol=1e-5):
    """
    Binary search for beta to match target perplexity.
    """
    logU = np.log(target_perplexity)
    beta = 1.0
    betamin = -np.inf
    betamax = np.inf
    
    H, P = Hbeta(D_act, beta)
    Hdiff = H - logU
    tries = 0

    while np.abs(Hdiff) > tol and tries < 50:
        if Hdiff > 0:
            betamin = beta
            beta = (beta + betamax) / 2 if betamax != np.inf else beta * 2
        else:
            betamax = beta
            beta = (beta + betamin) / 2 if betamin != -np.inf else beta / 2
        
        H, P = Hbeta(D_act, beta)
        Hdiff = H - logU
        tries += 1
        
    return P

def shannon_entropy_nn_weighted(adata, batch_key, label_key, embed, n_neighbors=90, perplexity=30):
    """
    Shannon entropy weighted by distance-based Gaussian kernels (LISI-style).
    """
    # 1. Get Neighbors and Distances
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(adata.obsm[embed])
    distances, indices = nn.kneighbors(adata.obsm[embed])
    
    # Remove self-neighbor
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # Prep Labels
    batches = adata.obs[batch_key].astype("category").cat.codes.values
    labels = adata.obs[label_key].astype("category").cat.codes.values
    n_batches = len(np.unique(batches))
    n_labels = len(np.unique(labels))

    batch_entropies = []
    label_entropies = []

    for i in range(indices.shape[0]):
        # 2. Get Gaussian weights P for this cell's neighborhood
        P = get_weighted_probs(distances[i], perplexity)
        
        # Neighbor indices for this cell
        neighbor_idx = indices[i]
        
        # 3. Aggregate probabilities by category
        # Batch Entropy
        batch_probs = np.zeros(n_batches)
        for b_idx, weight in zip(batches[neighbor_idx], P):
            batch_probs[b_idx] += weight

        # Label Entropy
        label_probs = np.zeros(n_labels)
        for l_idx, weight in zip(labels[neighbor_idx], P):
            label_probs[l_idx] += weight

        # 4. Compute Shannon Entropy (Normalized)
        # We use base 2 for consistency with your original snippet
        h_batch = entropy(batch_probs, base=2) / np.log2(n_batches) if n_batches > 1 else 0
        h_label = entropy(label_probs, base=2) / np.log2(n_labels) if n_labels > 1 else 0
        
        batch_entropies.append(h_batch)
        label_entropies.append(h_label)

    return np.mean(batch_entropies), 1 - np.mean(label_entropies)
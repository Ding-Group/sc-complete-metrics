from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scipy.stats import trim_mean
from tqdm import tqdm

from ..utils import check_adata, check_batch


def fisher_poisson(m1, m2):
    return np.sum( 2 * np.abs(np.sqrt(m1) - np.sqrt(m2)) )


def fisher_multinomial(m1, m2):
    v = np.sum(np.sqrt(m1 * m2))
    v = np.clip(v, -1, 1)
    return 2 * np.arccos(v)


def scgraph(
    adata,
    batch_key,
    label_key,
    embed,
    method: Literal['default', 'fisher_poisson', 'fisher_multinomial'] = 'default',
    metric: str = "euclidean",
    trim_rate: float = 0.05,
    thres_batch: int = 100,
    thres_celltype: int = 10,
    pca_comps: int = 10
):
    """scGraph

    :param adata: AnnData object with integrated embedding
    :param batch_key: Column in ``adata.obs`` with batch information
    :param label_key: Column in ``adata.obs`` with cell type information
    :param embed: Key in ``adata.obsm`` with integrated embedding
    :param method: Which version of SCGRAPH to use
    :param metric: Distance metric to use for calculating distances in the integrated
        embedding. If ``method`` is default, ``metric`` also refers to the distance
        metric in PCA space
    :param trim_rate: Trim rate for trimmed mean
    :param thres_batch: Minimum number of cells in a batch for the batch to be 
        included in groundtruth distances
    :param thres_celltype: Minimum number of cells for the cell type to be
        included in SCGRAPH computation
    """
    check_adata(adata)
    check_batch(batch_key, adata.obs)
    check_batch(label_key, adata.obs)

    scGraph_ = scGraph(
        adata,
        batch_key,
        label_key,
        method=method,
        metric=metric,
        trim_rate=trim_rate,
        thres_batch=thres_batch,
        thres_celltype=thres_celltype,
        pca_comps=pca_comps
    )
    res_df = scGraph_.main([embed])
    return ((res_df['Rank-PCA'] + res_df['Corr-PCA'] + res_df['Corr-Weighted']) / 3).iloc[0], res_df['Rank-PCA'], res_df['Corr-PCA'], res_df['Corr-Weighted']


# Copied from https://github.com/Genentech/Islander/blob/main/src/scGraph.py and modified to directly take in AnnData object.
class scGraph:
    def __init__(
        self,
        adata,
        batch_key,
        label_key,
        method: Literal['default', 'fisher_poisson', 'fisher_multinomial'] = 'default',
        metric: str = 'euclidean',
        trim_rate=0.05,
        thres_batch=100,
        thres_celltype=10,
        pca_comps: int = 10
    ):
        self.trim_rate = trim_rate
        self.thres_batch = thres_batch
        self.thres_celltype = thres_celltype
        self.ignore_celltype = []  # List of cell types to be ignored
        self.method = method
        if method == 'fisher_multinomial':
            self.metric = fisher_multinomial
        elif method == 'fisher_poisson':
            self.metric = fisher_poisson
        self.dist_metric = metric
        self._collect = dict()
        self.consensus_df = None
        self.pca_comps = pca_comps

        self.adata = adata
        self.batch_key = batch_key
        self.label_key = label_key

    def preprocess(self):
        for celltype in self.adata.obs[self.label_key].unique():
            if (
                self.adata.obs[self.label_key].value_counts()[celltype]
                < self.thres_celltype
            ):
                print(
                    f"Skipped cell type {celltype}, due to < {self.thres_celltype} cells"
                )
                self.ignore_celltype.append(celltype)

    def process_batches(self):
        print("Processing batches, calculate centroids and pairwise distances")
        for BATCH_ in tqdm(self.adata.obs[self.batch_key].unique()):
            adata_batch = self.adata[self.adata.obs[self.batch_key] == BATCH_].copy()

            if len(adata_batch) < self.thres_batch:
                print(f"Skipped batch {BATCH_}, due to < {self.thres_batch} cells")
                continue

            # NOTE: make sure the adata.X is log1p transformed, otherwise do it here
            # sc.pp.normalize_per_cell(adata_batch, counts_per_cell_after=1e4)
            # sc.pp.log1p(adata_batch)

            sc.pp.highly_variable_genes(adata_batch, n_top_genes=1000)
            sc.pp.pca(adata_batch, n_comps=self.pca_comps, mask_var=adata_batch.var.highly_variable)

            # NOTE: make sure
            centroids_pca = calculate_trimmed_means(
                adata_batch.obsm["X_pca"],
                adata_batch.obs[self.label_key],
                trim_proportion=self.trim_rate,
                ignore_=self.ignore_celltype,
            )
            pca_pairdist = compute_classwise_distances(centroids_pca, metric=self.dist_metric)
            self._collect[BATCH_] = pca_pairdist.div(
                pca_pairdist.max(axis=0), axis=1
            )

    def process_batches_fisher(self):
        print("Processing batches using Fisher information metric and pairwise distances")
        
        # Calculate highly variable genes if not already present
        if 'highly_variable' not in self.adata.var.columns:
            sc.pp.highly_variable_genes(self.adata)
        for BATCH_ in tqdm(self.adata.obs[self.batch_key].unique()):
            adata_batch = self.adata[self.adata.obs[self.batch_key] == BATCH_].copy()

            # Subset to just the highly variable genes
            adata_batch = adata_batch[:, adata_batch.var['highly_variable']]

            if len(adata_batch) < self.thres_batch:
                print(f"Skipped batch {BATCH_}, due to < {self.thres_batch} cells")
                continue

            if self.metric is fisher_poisson:
                means = compute_poisson_means(
                    adata_batch.layers['counts'],
                    adata_batch.obs[self.label_key],
                    ignore_=self.ignore_celltype
                )
            else:
                means = compute_multinomial_means(
                    adata_batch.layers['counts'],
                    adata_batch.obs[self.label_key],
                    ignore_=self.ignore_celltype
                )

            pairdist = compute_classwise_fisher(means, self.metric)
            self._collect[BATCH_] = pairdist.div(
                pairdist.max(axis=0), axis=1
            )

    def calculate_consensus(self):
        df_combined = pd.concat(self._collect.values(), axis=0, sort=False)
        self.consensus_df = df_combined.groupby(df_combined.index).mean()
        self.consensus_df = self.consensus_df.loc[self.consensus_df.columns, :]
        self.consensus_df = self.consensus_df / self.consensus_df.max(axis=0)

        # In case of numerical instability, set the diagonal of the DF to 0
        for i in range(len(self.consensus_df)):
            self.consensus_df.iloc[i, i] = 0

    @staticmethod
    def rank_diff(df1, df2):
        spearman_corr = {}
        for col in df1.columns:
            paired_non_nan = pd.concat([df1[col], df2[col]], axis=1).dropna()
            spearman_corr[col] = paired_non_nan.iloc[:, 0].corr(
                paired_non_nan.iloc[:, 1], method="spearman"
            )
        return pd.DataFrame.from_dict(
            spearman_corr, orient="index", columns=["Spearman Correlation"]
        )

    @staticmethod
    def corr_diff(df1, df2):
        pearson_corr = {}
        for col in df1.columns:
            paired_non_nan = pd.concat([df1[col], df2[col]], axis=1).dropna()
            pearson_corr[col] = paired_non_nan.iloc[:, 0].corr(
                paired_non_nan.iloc[:, 1], method="pearson"
            )
        return pd.DataFrame.from_dict(
            pearson_corr, orient="index", columns=["Pearson Correlation"]
        )

    @staticmethod
    def corrw_diff(df1, df2):
        pearson_corr = {}
        for col in df1.columns:
            paired_non_nan = pd.concat([df1[col], df2[col]], axis=1).dropna()
            pearson_corr[col] = scGraph.weighted_pearson(
                paired_non_nan.iloc[:, 0],
                paired_non_nan.iloc[:, 1],
                paired_non_nan.iloc[:, 1],
            )
        return pd.DataFrame.from_dict(
            pearson_corr, orient="index", columns=["Pearson Correlation"]
        )

    @staticmethod
    def weighted_pearson(x, y, distances):
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = 1 / distances
            weights[distances == 0] = 0
        weights /= np.sum(weights)
        weighted_mean_x = np.average(x, weights=weights)
        weighted_mean_y = np.average(y, weights=weights)
        covariance = np.sum(weights * (x - weighted_mean_x) * (y - weighted_mean_y))
        variance_x = np.sum(weights * (x - weighted_mean_x) ** 2)
        variance_y = np.sum(weights * (y - weighted_mean_y) ** 2)
        weighted_pearson_corr = covariance / np.sqrt(variance_x * variance_y)
        return weighted_pearson_corr

    def adata_consensus(self, obsm):
        _centroid = calculate_trimmed_means(
            np.array(self.adata.obsm[obsm]),
            self.adata.obs[self.label_key],
            trim_proportion=self.trim_rate,
            ignore_=self.ignore_celltype,
        )
        _pairdist = compute_classwise_distances(_centroid, metric=self.dist_metric)
        return _pairdist.div(_pairdist.max(axis=0), axis=1)

    def main(self, _obsm_list=None):
        self.preprocess()
        if self.method == 'default':
            self.process_batches()
        else:
            self.process_batches_fisher()
        self.calculate_consensus()

        res_df = pd.DataFrame(columns=["Rank-PCA", "Corr-PCA", "Corr-Weighted"])
        if _obsm_list is None:
            _obsm_list = sorted(list(self.adata.obsm))

        # self.consensus_df.to_csv("consensus_df_%s.csv"%self.trim_rate)
        # exit()
        for _obsm in _obsm_list:
            adata_df = self.adata_consensus(_obsm)
            _row_df = pd.DataFrame(
                {
                    "Rank-PCA": self.rank_diff(adata_df, self.consensus_df)
                    .mean()
                    .values,
                    "Corr-PCA": self.corr_diff(adata_df, self.consensus_df)
                    .mean()
                    .values,
                    "Corr-Weighted": self.corrw_diff(adata_df, self.consensus_df)
                    .mean()
                    .values,
                },
                index=[_obsm],
            )
            res_df = pd.concat([res_df, _row_df], axis=0, sort=False)
        return res_df


# Copied from https://github.com/Genentech/Islander/blob/main/src/Utils_Handler.py
def calculate_trimmed_means(X, labels, trim_proportion=0.2, ignore_=[]):
    centroids = dict()
    if isinstance(X, csr_matrix):
        X = X.toarray()
    for label in labels.unique():
        if label in ignore_:
            continue
        centroids[label] = trim_mean(
            X[labels == label], proportiontocut=trim_proportion, axis=0
        )
    return centroids


# Copied from https://github.com/Genentech/Islander/blob/main/src/Utils_Handler.py
def compute_classwise_distances(centroids, metric: str = "euclidean"):
    centroid_vectors = np.array([centroids[key] for key in sorted(centroids.keys())])
    if metric == 'l1':
        metric = 'cityblock'
    distances = cdist(centroid_vectors, centroid_vectors, metric)
    return pd.DataFrame(
        distances, columns=sorted(centroids.keys()), index=sorted(centroids.keys())
    )


def compute_poisson_means(X, labels, ignore_=[]):
    means = dict()
    if isinstance(X, csr_matrix):
        X = X.toarray()
    for label in labels.unique():
        if label in ignore_:
            continue
        means[label] = X[labels == label].mean(0)
    return means


def compute_multinomial_means(X, labels, ignore_=[]):
    means = dict()
    if isinstance(X, csr_matrix):
        X = X.toarray()
    for label in labels.unique():
        if label in ignore_:
            continue

        m = X[labels == label].sum(0)
        prop = m / np.sum(m)
        means[label] = prop
    return means


def compute_classwise_fisher(means, metric):
    mean_vectors = np.array([means[key] for key in sorted(means.keys())])
    distances = cdist(mean_vectors, mean_vectors, metric)
    return pd.DataFrame(
        distances, columns=sorted(means.keys()), index=sorted(means.keys())
    )

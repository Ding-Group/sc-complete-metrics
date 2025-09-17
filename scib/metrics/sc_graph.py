import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scipy.stats import trim_mean
from tqdm import tqdm

from ..utils import check_adata, check_batch


def scgraph(
    adata, batch_key, label_key, embed, trim_rate=0.05, thres_batch=100, thres_celltype=10
):
    """scGraph

    :param adata: AnnData object with integrated embedding

    """
    check_adata(adata)
    check_batch(batch_key, adata.obs)
    check_batch(label_key, adata.obs)

    scGraph_ = scGraph(
        adata,
        batch_key,
        label_key,
        trim_rate=trim_rate,
        thres_batch=thres_batch,
        thres_celltype=thres_celltype,
    )
    res_df = scGraph_.main([embed])
    return (res_df['Rank-PCA'] + res_df['Corr-PCA'] + res_df['Corr-Weighted']) / 3, res_df['Rank-PCA'], res_df['Corr-PCA'], res_df['Corr-Weighted']


# Copied from https://github.com/Genentech/Islander/blob/main/src/scGraph.py and modified to directly take in AnnData object.
class scGraph:
    def __init__(
        self,
        adata,
        batch_key,
        label_key,
        trim_rate=0.05,
        thres_batch=100,
        thres_celltype=10,
    ):
        self.trim_rate = trim_rate
        self.thres_batch = thres_batch
        self.thres_celltype = thres_celltype
        self.ignore_celltype = []  # List of cell types to be ignored
        self._collect_pca_ = dict()
        self.concensus_df_pca = None

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
        print("Processing batches, calcualte centroids and pairwise distances")
        for BATCH_ in tqdm(self.adata.obs[self.batch_key].unique()):
            adata_batch = self.adata[self.adata.obs[self.batch_key] == BATCH_].copy()

            if len(adata_batch) < self.thres_batch:
                print(f"Skipped batch {BATCH_}, due to < {self.thres_batch} cells")
                continue

            # NOTE: make sure the adata.X is log1p transformed, otherwise do it here
            # sc.pp.normalize_per_cell(adata_batch, counts_per_cell_after=1e4)
            # sc.pp.log1p(adata_batch)

            sc.pp.highly_variable_genes(adata_batch, n_top_genes=1000)
            sc.pp.pca(adata_batch, n_comps=10, mask_var=adata_batch.var.highly_variable)

            # NOTE: make sure
            centroids_pca = calculate_trimmed_means(
                adata_batch.obsm["X_pca"],
                adata_batch.obs[self.label_key],
                trim_proportion=self.trim_rate,
                ignore_=self.ignore_celltype,
            )
            pca_pairdist = compute_classwise_distances(centroids_pca)
            self._collect_pca_[BATCH_] = pca_pairdist.div(
                pca_pairdist.max(axis=0), axis=1
            )

    def calculate_consensus(self):
        df_combined = pd.concat(self._collect_pca_.values(), axis=0, sort=False)
        self.concensus_df_pca = df_combined.groupby(df_combined.index).mean()
        self.concensus_df_pca = self.concensus_df_pca.loc[
            self.concensus_df_pca.columns, :
        ]
        self.concensus_df_pca = self.concensus_df_pca / self.concensus_df_pca.max(
            axis=0
        )

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

    def adata_concensus(self, obsm):
        _centroid = calculate_trimmed_means(
            np.array(self.adata.obsm[obsm]),
            self.adata.obs[self.label_key],
            trim_proportion=self.trim_rate,
            ignore_=self.ignore_celltype,
        )
        _pairdist = compute_classwise_distances(_centroid)
        return _pairdist.div(_pairdist.max(axis=0), axis=1)

    def main(self, _obsm_list=None):
        self.preprocess()
        self.process_batches()
        self.calculate_consensus()

        res_df = pd.DataFrame(columns=["Rank-PCA", "Corr-PCA", "Corr-Weighted"])
        if _obsm_list is None:
            _obsm_list = sorted(list(self.adata.obsm))

        # self.concensus_df_pca.to_csv("concensus_df_pca_%s.csv"%self.trim_rate)
        # exit()
        for _obsm in _obsm_list:
            adata_df = self.adata_concensus(_obsm)
            _row_df = pd.DataFrame(
                {
                    "Rank-PCA": self.rank_diff(adata_df, self.concensus_df_pca)
                    .mean()
                    .values,
                    "Corr-PCA": self.corr_diff(adata_df, self.concensus_df_pca)
                    .mean()
                    .values,
                    "Corr-Weighted": self.corrw_diff(adata_df, self.concensus_df_pca)
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
def compute_classwise_distances(centroids):
    centroid_vectors = np.array([centroids[key] for key in sorted(centroids.keys())])
    distances = cdist(centroid_vectors, centroid_vectors, "euclidean")
    return pd.DataFrame(
        distances, columns=sorted(centroids.keys()), index=sorted(centroids.keys())
    )

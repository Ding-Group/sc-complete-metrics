#!/usr/bin/env python3

import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from anndata._io.utils import OldFormatWarning
    warnings.filterwarnings("ignore", category=OldFormatWarning)
except ImportError:
    pass

import os
import sys
import time
import argparse

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

import scanpy as sc
sc.settings.verbosity = 'error'
import pandas as pd

# 1) 
from scib.metrics.metrics import metrics

# 2) single matrics
from scib.metrics.calinski_harabasz     import calinski_harabasz
from scib.metrics.completeness           import completeness
from scib.metrics.homogeneity            import homogeneity
from scib.metrics.v_measure              import v_measure
from scib.metrics.compactness            import compactness
from scib.metrics.modularity             import modularity
from scib.metrics.crossfold_logistic     import crossfold_logistic
from scib.metrics.batch_probing_score    import batch_probing_score
from scib.metrics.fowlkes_mallows        import fowlkes_mallows
from scib.metrics.separation             import separation
from scib.metrics.kl_divergence          import kl_divergence
from scib.metrics.over_correction        import over_correction
from scib.metrics.shannon_entropy        import shannon_entropy
from scib.metrics.highly_variable_genes  import hvg_overlap
from scib.metrics.silhouette             import silhouette

from scib.metrics.trajectory             import trajectory_conservation as trajectory
from scib.metrics.balanced_metrics       import balanced_accuracy as balanced_metrics
from scib.metrics.agreement              import agreement
from scib.metrics.knn_divergence         import knn_divergence
from scib.metrics.cluster_batch_entropy  import cluster_batch_entropy
from scib.metrics.morans_i               import morans_i
from scib.metrics.cell_cycle             import cell_cycle
from scib.metrics.pcr                    import pcr
from scib.metrics.f1_score               import f1_score_macro as f1_score
from scib.metrics.clustering_accuracy    import clustering_accuracy
from scib.metrics.purity                 import purity_score as purity
from scib.metrics.jaccard                import jaccard_index as jaccard
from scib.metrics.ami                    import ami
from scib.metrics.rbet                   import rbet
from scib.metrics.alignment              import alignment
from scib.metrics.sc_graph               import sc_graph_ as sc_graph_func
from scib.metrics.matthews_corrcoef      import matthews_corrcoef
from scib.metrics.kbet                   import kBET
from scib.metrics.crossfold_knn          import crossfold_knn
from scib.metrics.ari                    import ari
from scib.metrics.isolated_labels        import isolated_labels
from scib.metrics.coefficient_variation  import coefficient_of_variation as coefficient_variation

def main():
    parser = argparse.ArgumentParser(
        description="Run scIB metrics plus every single-metric module on an AnnData dataset"
    )
    parser.add_argument("-a", "--adata", required=True,
                        help="Path to input AnnData .h5ad file")
    args = parser.parse_args()

    print("=== Starting run_test ===")
    print(f"Loading AnnData from {args.adata} …")
    adata = sc.read_h5ad(args.adata)
    adata.obs['batch']    = adata.obs['study'].astype('category')
    adata.obs['celltype'] = adata.obs['cell_type'].astype('category')

    print("Running PCA …")
    sc.pp.pca(adata, n_comps=50, svd_solver='arpack')

    print("Building neighbor graph & Leiden clustering …")
    sc.pp.neighbors(adata, use_rep='X_pca', n_pcs=50)
    sc.tl.leiden(adata, key_added='cluster', resolution=1.0)
    adata.obs['cluster'] = adata.obs['cluster'].astype('category')

    adata_int = adata.copy()
    adata_int.obsm['X_pca'] = adata.obsm['X_pca']
    print("Computing UMAP …")
    sc.tl.umap(adata_int)  

    # 1) scIB metrics
    print("Computing scIB native metrics …")
    scores_df = metrics(
        adata, adata_int,
        batch_key='batch', label_key='celltype', cluster_key='cluster',
        hvg_score_=True, silhouette_=True, ari_=True, nmi_=True,
        graph_conn_=True, pcr_=True, ilisi_=True, clisi_=True,
        kBET_=False, sc_graph_=True,
        # if use trajectory -> True
        trajectory_=False,
        isolated_labels_=True, isolated_labels_f1_=True, isolated_labels_asw_=True,
        subsample=0.5, n_cores=1, verbose=False
    )
    if isinstance(scores_df, pd.DataFrame):
        scores_series = scores_df.iloc[:, 0]
    else:
        scores_series = scores_df
    base_scores = scores_series.to_dict()
    print("  scIB metrics computed, keys:", list(base_scores.keys()))

    # 2) more 
    print("Computing *every* single-metric module …")

    # print("DEBUG adata_int.uns keys:", list(adata_int.uns.keys()))
    # if "neighbors" in adata_int.uns:
    #     print("DEBUG adata_int.uns['neighbors'] keys:", list(adata_int.uns["neighbors"].keys()))
    # print("DEBUG adata_int.obsp keys:", list(adata_int.obsp.keys()))

    single_metrics = {
        'calinski_harabasz':     lambda: calinski_harabasz(adata, cluster_key='cluster', representation='X_pca'),
        'completeness':          lambda: completeness(adata, cluster_key='cluster', label_key='celltype'),
        'homogeneity':           lambda: homogeneity(adata, cluster_key='cluster', label_key='celltype'),
        'v_measure':             lambda: v_measure(adata, cluster_key='cluster', label_key='celltype'),
        'compactness':           lambda: compactness(adata, cluster_key='cluster', label_key='celltype'),
        'modularity':            lambda: modularity(adata_int, cluster_key='cluster'),
        'crossfold_logistic':    lambda: crossfold_logistic(adata, adata_int, cluster_key='cluster', label_key='celltype'),
        'batch_probing_score':   lambda: batch_probing_score(adata, cluster_key='cluster', batch_key='batch'),
        'fowlkes_mallows':       lambda: fowlkes_mallows(adata, cluster_key='cluster', label_key='celltype'),
        'separation':            lambda: separation(adata, cluster_key='cluster', label_key='celltype'),
        'kl_divergence':         lambda: kl_divergence(adata, cluster_key='cluster', batch_key='batch'),
        'over_correction':       lambda: over_correction(adata, adata_int, batch_key='batch', label_key='celltype'),
        'shannon_entropy':       lambda: shannon_entropy(adata, cluster_key='cluster', batch_key='batch'),
        'hvg_overlap':           lambda: hvg_overlap(adata, adata_int, batch_key='batch'),
        'silhouette':            lambda: silhouette(adata_int,label_key='celltype', embed='X_pca'),
        'trajectory':            lambda: trajectory(adata, adata_int, batch_key='batch', label_key='celltype'),
        'balanced_metrics':      lambda: balanced_metrics(adata, cluster_key='cluster', label_key='celltype'),
        'agreement':             lambda: agreement(adata, cluster_key='cluster', label_key='celltype'),
        'knn_divergence':        lambda: knn_divergence(adata_int, batch_key='batch', embed='X_pca', k=30),
        'cluster_batch_entropy': lambda: cluster_batch_entropy(adata, batch_key='batch', cluster_key='cluster'),
        'morans_i':              lambda: morans_i(adata, 'X_pca', label_key='celltype'),
        'cell_cycle':            lambda: cell_cycle(adata, cell_cycle_key='phase'),
        'pcr':                   lambda: pcr(adata, covariate='batch'),
        'f1_score':              lambda: f1_score(adata, cluster_key='cluster', label_key='celltype'),
        'clustering_accuracy':   lambda: clustering_accuracy(adata,'cluster','celltype'),
        'purity':                lambda: purity(adata, cluster_key='cluster', label_key='celltype'),
        'jaccard':               lambda: jaccard(adata, cluster_key='cluster', label_key='celltype'),
        'ami':                   lambda: ami(adata, cluster_key='cluster', label_key='celltype'),
        'rbet':                  lambda: rbet(adata, batch_key='batch',label_key='celltype'),
        'alignment':             lambda: alignment(adata_int, batch_key='batch',embed='X_umap'),
        'sc_graph':              lambda: sc_graph_func(adata_int, batch_key='batch'),
        'matthews_corrcoef':     lambda: matthews_corrcoef(adata, cluster_key='cluster', label_key='celltype'),
        'kBET':                  lambda: kBET(adata_int, batch_key='batch', label_key='celltype',type_='knn'),
        'crossfold_knn':         lambda: crossfold_knn(adata, cluster_key='cluster', label_key='celltype'),
        'ARI':                   lambda: ari(adata, cluster_key='cluster', label_key='celltype'),
        'isolated_labels':       lambda: isolated_labels(adata, batch_key='batch', label_key='celltype',embed='X_pca'),
        'coefficient_variation': lambda: coefficient_variation(adata, gene_list=adata.var_names),
    }

    extra_scores = {}
    for name, fn in single_metrics.items():
        print(f"→ Running {name} …", end="", flush=True)
        t0 = time.time()
        try:
            val = fn()
            dt = time.time() - t0
            print(f" done in {dt:.1f}s")
            extra_scores[name] = val
        except Exception as e:
            print(f" FAILED ({e.__class__.__name__})")

    # 3) summery and print
    print("\nAll scores:")
    all_scores = {**base_scores, **extra_scores}
    for name, val in all_scores.items():
        try:
            print(f"{name:30s} {val:.6f}")
        except:
            print(f"{name:30s} {val}")

    print("=== Done ===")


if __name__ == "__main__":
    main()

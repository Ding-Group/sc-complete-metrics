# #!/usr/bin/env python3

# import warnings
# warnings.simplefilter("ignore")
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=UserWarning)

# try:
#     from anndata._io.specs.registry import OldFormatWarning
#     warnings.filterwarnings("ignore", category=OldFormatWarning)
# except ImportError:
#     pass

# warnings.filterwarnings(
#     "ignore",
#     message=r"Element '.*/.*' was written without encoding metadata"
# )


# import os
# import sys
# import time
# import argparse

# HERE = os.path.dirname(__file__)
# ROOT = os.path.abspath(os.path.join(HERE, ".."))
# sys.path.insert(0, ROOT)

# import scanpy as sc
# sc.settings.verbosity = 'error'
# import pandas as pd

# # 1) 
# from scib.metrics.metrics import metrics

# # 2) single matrics
# from scib.metrics.calinski_harabasz     import calinski_harabasz
# from scib.metrics.completeness           import completeness
# from scib.metrics.homogeneity            import homogeneity
# from scib.metrics.v_measure              import v_measure
# from scib.metrics.compactness            import compactness
# from scib.metrics.modularity             import modularity
# from scib.metrics.crossfold_logistic     import crossfold_logistic
# from scib.metrics.batch_probing_score    import batch_probing_score
# from scib.metrics.fowlkes_mallows        import fowlkes_mallows
# from scib.metrics.separation             import separation
# from scib.metrics.kl_divergence          import kl_divergence
# from scib.metrics.over_correction        import over_correction
# from scib.metrics.shannon_entropy        import shannon_entropy
# from scib.metrics.highly_variable_genes  import hvg_overlap
# from scib.metrics.silhouette             import silhouette

# from scib.metrics.trajectory             import trajectory_conservation as trajectory
# from scib.metrics.balanced_metrics       import balanced_accuracy as balanced_metrics
# from scib.metrics.agreement              import agreement
# from scib.metrics.knn_divergence         import knn_divergence
# from scib.metrics.cluster_batch_entropy  import cluster_batch_entropy
# from scib.metrics.morans_i               import morans_i
# from scib.metrics.cell_cycle             import cell_cycle
# from scib.metrics.pcr                    import pcr
# from scib.metrics.f1_score               import f1_score_macro as f1_score
# from scib.metrics.clustering_accuracy    import clustering_accuracy
# from scib.metrics.purity                 import purity_score as purity
# from scib.metrics.jaccard                import jaccard_index as jaccard
# from scib.metrics.ami                    import ami
# from scib.metrics.rbet                   import rbet
# from scib.metrics.alignment              import alignment
# from scib.metrics.sc_graph               import sc_graph_ as sc_graph_func
# from scib.metrics.matthews_corrcoef      import matthews_corrcoef
# from scib.metrics.kbet                   import kBET
# from scib.metrics.crossfold_knn          import crossfold_knn
# from scib.metrics.ari                    import ari
# from scib.metrics.isolated_labels        import isolated_labels
# from scib.metrics.coefficient_variation  import coefficient_of_variation as coefficient_variation
# from scib.metrics.ldf_diff               import ldf_diff
# from scib.metrics.cell_specific_mixing_score import cell_specific_mixing_score
# from scib.metrics.crossbatch_knn         import crossbatch_knn
# from scib.metrics.crossbatch_logistic    import crossbatch_logistic
# from scib.metrics.batch_ari              import batch_ari
# from scib.metrics.batch_purity_score     import batch_purity_score
# from scib.metrics.batch_shannon_entropy  import batch_shannon_entropy





# def main():
#     parser = argparse.ArgumentParser(
#         description="Run scIB metrics plus every single-metric module on an AnnData dataset"
#     )
#     parser.add_argument("-a", "--adata", required=True,
#                         help="Path to input AnnData .h5ad file")
#     parser.add_argument(
#         "--batch_key",
#         dest="batch_key",
#         default="study",      
#         help="Column in adata.obs to use as batch labels"
#     )
#     parser.add_argument(
#         "--label_key",
#         dest="label_key",
#         default="cell_type",  
#         help="Column in adata.obs to use as cell-type labels"
#     )
#     args = parser.parse_args()

#     print("=== Starting run_test ===")
#     print(f"Loading AnnData from {args.adata} …")
#     adata = sc.read_h5ad(args.adata)
#     # adata.obs['batch']    = adata.obs['study'].astype('category')
#     # adata.obs['celltype'] = adata.obs['cell_type'].astype('category')
#     adata.obs['batch']    = adata.obs[args.batch_key].astype('category')
#     adata.obs['celltype'] = adata.obs[args.label_key].astype('category')


#     print("Running PCA …")
#     sc.pp.pca(adata, n_comps=50, svd_solver='arpack')

#     print("Building neighbor graph & Leiden clustering …")
#     sc.pp.neighbors(adata, use_rep='X_pca', n_pcs=50)
#     sc.tl.leiden(adata, key_added='cluster', resolution=1.0)
#     adata.obs['cluster'] = adata.obs['cluster'].astype('category')

#     adata_int = adata.copy()
#     adata_int.obsm['X_pca'] = adata.obsm['X_pca']
#     print("Computing UMAP …")
#     sc.tl.umap(adata_int)  

#     # 1) scIB metrics
#     print("Computing scIB native metrics …")
#     scores_df = metrics(
#         adata, adata_int,
#         batch_key=args.batch_key, label_key=args.label_key, cluster_key='cluster',

#         hvg_score_=True, silhouette_=True, ari_=True, nmi_=True,
#         graph_conn_=True, pcr_=True, ilisi_=True, clisi_=True,
#         kBET_=False, sc_graph_=True,alignment_=True,
#         # if use trajectory -> True
#         trajectory_=False,
#         isolated_labels_=True, isolated_labels_f1_=True, isolated_labels_asw_=True,
#         subsample=0.5, n_cores=1, verbose=False,
#         crossbatch_logistic_=True,
#         crossbatch_logistic_scoring="accuracy", 
#         batch_ari_=True,
#         batch_purity_=True, 
#         **{"batch_shannon_entropy_": True},
#     )
#     if isinstance(scores_df, pd.DataFrame):
#         scores_series = scores_df.iloc[:, 0]
#     else:
#         scores_series = scores_df
#     base_scores = scores_series.to_dict()
#     print("  scIB metrics computed, keys:", list(base_scores.keys()))

#     # 2) more 
#     print("Computing *every* single-metric module …")

#     single_metrics = {
#         'calinski_harabasz':     lambda: calinski_harabasz(adata, cluster_key='cluster', representation='X_pca'),
#         'completeness':          lambda: completeness(adata, cluster_key='cluster', label_key=args.label_key),
#         'homogeneity':           lambda: homogeneity(adata, cluster_key='cluster', label_key=args.label_key),
#         'v_measure':             lambda: v_measure(adata, cluster_key='cluster', label_key=args.label_key),
#         'compactness':           lambda: compactness(adata, cluster_key='cluster', label_key=args.label_key),
#         'modularity':            lambda: modularity(adata_int, cluster_key='cluster'),
#         'crossfold_logistic':    lambda: crossfold_logistic(adata, adata_int, cluster_key='cluster', label_key=args.label_key),
#         'batch_probing_score':   lambda: batch_probing_score(adata, cluster_key='cluster', batch_key=args.batch_key),
#         'fowlkes_mallows':       lambda: fowlkes_mallows(adata, cluster_key='cluster', label_key=args.label_key),
#         'separation':            lambda: separation(adata, cluster_key='cluster', label_key=args.label_key),
#         'kl_divergence':         lambda: kl_divergence(adata, cluster_key='cluster', batch_key=args.batch_key),
#         'over_correction':       lambda: over_correction(adata, adata_int, batch_key=args.batch_key, label_key=args.label_key),
#         'shannon_entropy':       lambda: shannon_entropy(adata, cluster_key='cluster', batch_key=args.batch_key),
#         #'hvg_overlap':           lambda: hvg_overlap(adata, adata_int, batch_key=args.batch_key),
#         #'silhouette':            lambda: silhouette(adata_int, label_key=args.label_key, embed='X_pca'),
#         'trajectory':            lambda: trajectory(adata, adata_int, batch_key=args.batch_key, label_key=args.label_key),
#         'balanced_metrics':      lambda: balanced_metrics(adata, cluster_key='cluster', label_key=args.label_key),
#         'agreement':             lambda: agreement(adata, cluster_key='cluster', label_key=args.label_key),
#         'knn_divergence':        lambda: knn_divergence(adata_int, batch_key=args.batch_key, embed='X_pca', k=30),
#         'cluster_batch_entropy': lambda: cluster_batch_entropy(adata, batch_key=args.batch_key, cluster_key='cluster'),
#         'morans_i':              lambda: morans_i(adata, 'X_pca', label_key=args.label_key),
#         'cell_cycle_conservation':            lambda: cell_cycle(adata, cell_cycle_key='phase'),
#         #'pcr':                   lambda: pcr(adata, covariate=args.batch_key),
#         'f1_score':              lambda: f1_score(adata, cluster_key='cluster', label_key=args.label_key),
#         'clustering_accuracy':   lambda: clustering_accuracy(adata, cluster_key='cluster', label_key=args.label_key),
#         'purity':                lambda: purity(adata, cluster_key='cluster', label_key=args.label_key),
#         'jaccard':               lambda: jaccard(adata, cluster_key='cluster', label_key=args.label_key),
#         'ami':                   lambda: ami(adata, cluster_key='cluster', label_key=args.label_key),
#         'rbet':                  lambda: rbet(adata, batch_key=args.batch_key, label_key=args.label_key),
#         #'alignment':             lambda: alignment(adata_int, batch_key=args.batch_key, embed='X_umap'),
#         #'sc_graph':              lambda: sc_graph_func(adata_int, batch_key=args.batch_key),
#         'matthews_corrcoef':     lambda: matthews_corrcoef(adata, cluster_key='cluster', label_key=args.label_key),
#         'kBET':                  lambda: kBET(adata_int, batch_key=args.batch_key, label_key=args.label_key, type_='knn'),
#         'crossfold_knn':         lambda: crossfold_knn(adata, cluster_key='cluster', label_key=args.label_key),
#         #'ARI':                   lambda: ari(adata, cluster_key='cluster', label_key=args.label_key),
#         #'isolated_labels':       lambda: isolated_labels(adata, batch_key=args.batch_key, label_key=args.label_key, embed='X_pca'),
#         'coefficient_variation': lambda: coefficient_variation(adata, gene_list=adata.var_names),
#         'ldf_diff':              lambda: ldf_diff(adata, batch_key=args.batch_key, embed='X_pca'),
#         'cell_specific_mixing_score': lambda: cell_specific_mixing_score(adata, batch_key=args.batch_key, embed='X_pca'),
#         'crossbatch_knn':        lambda: crossbatch_knn(adata_int, batch_key=args.batch_key, label_key=args.label_key, embed="X_pca"),
#         #'crossbatch_logistic': lambda: crossbatch_logistic(adata_int, label_key=args.label_key, batch_key=args.batch_key, use_rep='X_pca', scoring='accuracy', max_iter=1000),
#         'batch_ari': lambda: batch_ari(adata_int, cluster_key='cluster', batch_key=args.batch_key),
#         'batch_purity_score': lambda: batch_purity_score(adata, cluster_key='cluster', batch_key=args.batch_key),
#         'batch_shannon_entropy': (lambda: batch_shannon_entropy(adata_int, cluster_key='cluster', batch_key=args.batch_key, normalize=True)),
#     }




#     extra_scores = {}
#     for name, fn in single_metrics.items():
#         print(f"→ Running {name} …", end="", flush=True)
#         t0 = time.time()
#         try:
#             val = fn()
#             dt = time.time() - t0
#             print(f" done in {dt:.1f}s")
#             extra_scores[name] = val
#         except Exception as e:
#             print(f" FAILED ({e.__class__.__name__})")

#     import math

#     def _is_nan(x):
#         try:
#             return isinstance(x, float) and math.isnan(x)
#         except Exception:
#             return False
#     # 3) summery and print
#     print("\nAll scores:")

#     for k in list(extra_scores.keys()):
#         if k in base_scores:
#             if _is_nan(base_scores[k]) or base_scores[k] is None:
#                 base_scores[k] = extra_scores.pop(k)   
#             else:
#                 del extra_scores[k]                    


#     all_scores = {**base_scores, **extra_scores}
#     for name, val in all_scores.items():
#         try:
#             print(f"{name:30s} {val:.6f}")
#         except:
#             print(f"{name:30s} {val}")

#     print("=== Done ===")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from anndata._io.specs.registry import OldFormatWarning
    warnings.filterwarnings("ignore", category=OldFormatWarning)
except ImportError:
    pass
warnings.filterwarnings(
    "ignore",
    message=r"Element '.*/.*' was written without encoding metadata"
)

import os
import sys
import time
import argparse
import math

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

import scanpy as sc
sc.settings.verbosity = 'error'
import pandas as pd

# 1) scIB bundled metrics
from scib.metrics.metrics import metrics

# 2) single metrics
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
from scib.metrics.ldf_diff               import ldf_diff
from scib.metrics.cell_specific_mixing_score import cell_specific_mixing_score
from scib.metrics.crossbatch_knn         import crossbatch_knn
from scib.metrics.crossbatch_logistic    import crossbatch_logistic
from scib.metrics.batch_ari              import batch_ari
from scib.metrics.batch_purity_score     import batch_purity_score
from scib.metrics.batch_shannon_entropy  import batch_shannon_entropy


def _is_nan(x):
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run scIB metrics plus every single-metric module on an AnnData dataset"
    )
    parser.add_argument(
        "-a", "--adata", required=True,
        help="Path to input AnnData .h5ad file (raw/original)"
    )
    parser.add_argument(
        "--batch_key", dest="batch_key", default="study",
        help="Column in adata.obs to use as batch labels"
    )
    parser.add_argument(
        "--label_key", dest="label_key", default="cell_type",
        help="Column in adata.obs to use as cell-type labels"
    )
    # update here --------Aug 12:
    parser.add_argument(
        "--integrated_adata", dest="integrated_adata", default=None,
        help="Path to integrated AnnData .h5ad (if provided, used as adata_int)"
    )
    parser.add_argument(
        "--embed", dest="embed", default="X_pca",
        help="Embedding key to use for metrics (e.g. X_pca_harmony, X_scVI)"
    )
    parser.add_argument(
        "--out", dest="outdir", default=None,
        help="Output directory to save scores.csv (if None, only prints to stdout)"
    )

    args = parser.parse_args()

    print("=== Starting run_test ===")
    print(f"Loading raw AnnData from {args.adata} …")
    adata = sc.read_h5ad(args.adata)

    # Standardize obs keys on raw adata
    adata.obs['batch']    = adata.obs[args.batch_key].astype('category')
    adata.obs['celltype'] = adata.obs[args.label_key].astype('category')

    # Ensure PCA exists on raw (needed for baseline & some metrics)
    print("Running PCA on raw …")
    sc.pp.pca(adata, n_comps=50, svd_solver='arpack')

    # Build neighbors & Leiden on raw (cluster_key='cluster')
    print("Building neighbor graph & Leiden clustering on raw …")
    sc.pp.neighbors(adata, use_rep='X_pca', n_pcs=50)
    sc.tl.leiden(adata, key_added='cluster', resolution=1.0)
    adata.obs['cluster'] = adata.obs['cluster'].astype('category')

    # Decide adata_int
    if args.integrated_adata:
        print(f"Loading integrated AnnData from {args.integrated_adata} …")
        adata_int = sc.read_h5ad(args.integrated_adata)

        # Check index alignment & propagate batch/label if missing
        if not adata_int.obs_names.equals(adata.obs_names):
            raise ValueError("obs_names of integrated_adata and raw adata do not match; cannot safely copy labels.")
        if 'batch' not in adata_int.obs:
            adata_int.obs['batch'] = adata.obs['batch']
        if 'celltype' not in adata_int.obs:
            adata_int.obs['celltype'] = adata.obs['celltype']
    else:
        # Baseline: use copy of raw
        adata_int = adata.copy()
        # Ensure embed exists at least for X_pca
        if 'X_pca' in adata.obsm and 'X_pca' not in adata_int.obsm:
            adata_int.obsm['X_pca'] = adata.obsm['X_pca']
        # UMAP is only for optional metrics; keep previous behavior for baseline
        print("Computing UMAP on baseline integrated (copy of raw) …")
        sc.tl.umap(adata_int)

    # Validate embed key presence (warn or fail early)
    if args.embed not in adata_int.obsm:
        available = list(adata_int.obsm.keys())
        msg = (f"Requested embed '{args.embed}' not found in adata_int.obsm. "
               f"Available keys: {available}")
        
        raise KeyError(msg)

    # 1) scIB bundled metrics
    print("Computing scIB native metrics …")
    scores_df = metrics(
        adata, adata_int,
        batch_key=args.batch_key,
        label_key=args.label_key,
        cluster_key='cluster',
        embed=args.embed,

        hvg_score_=True, silhouette_=True, ari_=True, nmi_=True,
        graph_conn_=True, pcr_=True, ilisi_=True, clisi_=True,
        kBET_=True, sc_graph_=True, alignment_=True,
        # if use trajectory -> True
        trajectory_=False,
        isolated_labels_=True, isolated_labels_f1_=True, isolated_labels_asw_=True,
        subsample=0.5, n_cores=1, verbose=False,
        crossbatch_logistic_=True,
        crossbatch_logistic_scoring="accuracy",
        batch_ari_=True,
        batch_purity_=True,
        **{"batch_shannon_entropy_": True},
    )

    if isinstance(scores_df, pd.DataFrame):
        scores_series = scores_df.iloc[:, 0]
    else:
        scores_series = scores_df
    base_scores = scores_series.to_dict()
    print("  scIB metrics computed, keys:", list(base_scores.keys()))

    # 2) more single metrics
    print("Computing *every* single-metric module …")
    embed = args.embed  # shorthand

    single_metrics = {
        'calinski_harabasz':     lambda: calinski_harabasz(adata, cluster_key='cluster', representation=embed),
        'completeness':          lambda: completeness(adata, cluster_key='cluster', label_key=args.label_key),
        'homogeneity':           lambda: homogeneity(adata, cluster_key='cluster', label_key=args.label_key),
        'v_measure':             lambda: v_measure(adata, cluster_key='cluster', label_key=args.label_key),
        'compactness':           lambda: compactness(adata, cluster_key='cluster', label_key=args.label_key),
        'modularity':            lambda: modularity(adata_int, cluster_key='cluster'),
        'crossfold_logistic':    lambda: crossfold_logistic(adata, adata_int, cluster_key='cluster', label_key=args.label_key),
        'batch_probing_score':   lambda: batch_probing_score(adata, cluster_key='cluster', batch_key=args.batch_key),
        'fowlkes_mallows':       lambda: fowlkes_mallows(adata, cluster_key='cluster', label_key=args.label_key),
        'separation':            lambda: separation(adata, cluster_key='cluster', label_key=args.label_key),
        'kl_divergence':         lambda: kl_divergence(adata, cluster_key='cluster', batch_key=args.batch_key),
        'over_correction':       lambda: over_correction(adata, adata_int, batch_key=args.batch_key, label_key=args.label_key),
        'shannon_entropy':       lambda: shannon_entropy(adata, cluster_key='cluster', batch_key=args.batch_key),
        # 'hvg_overlap':         lambda: hvg_overlap(adata, adata_int, batch_key=args.batch_key),
        # 'silhouette':          lambda: silhouette(adata_int, label_key=args.label_key, embed=embed),
        'trajectory':            lambda: trajectory(adata, adata_int, batch_key=args.batch_key, label_key=args.label_key),
        'balanced_metrics':      lambda: balanced_metrics(adata, cluster_key='cluster', label_key=args.label_key),
        'agreement':             lambda: agreement(adata, cluster_key='cluster', label_key=args.label_key),
        'knn_divergence':        lambda: knn_divergence(adata_int, batch_key=args.batch_key, embed=embed, k=30),
        'cluster_batch_entropy': lambda: cluster_batch_entropy(adata, batch_key=args.batch_key, cluster_key='cluster'),
        'morans_i':              lambda: morans_i(adata, embed, label_key=args.label_key),
        'cell_cycle_conservation': lambda: cell_cycle(adata, cell_cycle_key='phase'),
        # 'pcr':                 lambda: pcr(adata, covariate=args.batch_key),
        'f1_score':              lambda: f1_score(adata, cluster_key='cluster', label_key=args.label_key),
        'clustering_accuracy':   lambda: clustering_accuracy(adata, cluster_key='cluster', label_key=args.label_key),
        'purity':                lambda: purity(adata, cluster_key='cluster', label_key=args.label_key),
        'jaccard':               lambda: jaccard(adata, cluster_key='cluster', label_key=args.label_key),
        'ami':                   lambda: ami(adata, cluster_key='cluster', label_key=args.label_key),
        'rbet':                  lambda: rbet(adata, batch_key=args.batch_key, label_key=args.label_key),
        # 'alignment':           lambda: alignment(adata_int, batch_key=args.batch_key, embed='X_umap'),
        # 'sc_graph':            lambda: sc_graph_func(adata_int, batch_key=args.batch_key),
        'matthews_corrcoef':     lambda: matthews_corrcoef(adata, cluster_key='cluster', label_key=args.label_key),
        'kBET':                  lambda: kBET(adata_int, batch_key=args.batch_key, label_key=args.label_key, type_='knn', embed=embed),
        'crossfold_knn':         lambda: crossfold_knn(adata, cluster_key='cluster', label_key=args.label_key),
        # 'ARI':                 lambda: ari(adata, cluster_key='cluster', label_key=args.label_key),
        # 'isolated_labels':     lambda: isolated_labels(adata, batch_key=args.batch_key, label_key=args.label_key, embed=embed),
        'coefficient_variation': lambda: coefficient_variation(adata, gene_list=adata.var_names),
        'ldf_diff':              lambda: ldf_diff(adata, batch_key=args.batch_key, embed=embed),
        'cell_specific_mixing_score': lambda: cell_specific_mixing_score(adata, batch_key=args.batch_key, embed=embed),
        'crossbatch_knn':        lambda: crossbatch_knn(adata_int, batch_key=args.batch_key, label_key=args.label_key, embed=embed),
        # 'crossbatch_logistic': lambda: crossbatch_logistic(adata_int, label_key=args.label_key, batch_key=args.batch_key, use_rep=embed, scoring='accuracy', max_iter=1000),
        'batch_ari':             lambda: batch_ari(adata_int, cluster_key='cluster', batch_key=args.batch_key),
        'batch_purity_score':    lambda: batch_purity_score(adata, cluster_key='cluster', batch_key=args.batch_key),
        'batch_shannon_entropy': lambda: batch_shannon_entropy(adata_int, cluster_key='cluster', batch_key=args.batch_key, normalize=True),
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

    # 3) summary & print
    print("\nAll scores:")

    # Merge: if base has NaN, backfill from single metric; else prefer base
    for k in list(extra_scores.keys()):
        if k in base_scores:
            if _is_nan(base_scores[k]) or base_scores[k] is None:
                base_scores[k] = extra_scores.pop(k)
            else:
                del extra_scores[k]

    all_scores = {**base_scores, **extra_scores}
    for name, val in all_scores.items():
        try:
            print(f"{name:30s} {val:.6f}")
        except Exception:
            print(f"{name:30s} {val}")

    # ---- save to disk if requested
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        out_csv = os.path.join(args.outdir, "scores.csv")
        df = pd.DataFrame(list(all_scores.items()), columns=["metric", "value"])
        df.to_csv(out_csv, index=False)
        print(f"[saved] {out_csv}")

    print("=== Done ===")


if __name__ == "__main__":
    main()

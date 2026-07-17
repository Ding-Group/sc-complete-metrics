import inspect
import math

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


from ..utils import check_adata, check_batch
from .adgbd import adgbd
from .adjusted_wallace_score import adjusted_wallace_score
from .agreement import agreement
from .alignment import alignment
from .ami import ami
from .ari import ari, ari_fixed
from .bAMI import bAMI
from .bARI import bARI
from .batch_ari import batch_ari
from .batch_probing_score import batch_probing_score, per_label_batch_probing_score
from .bras import bras
from .bv_measure import bv_measure
from .calinski_harabasz import calinski_harabasz
from .cell_cycle import cell_cycle
from .cell_specific_mixing_score import cell_specific_mixing_score
from .cilisi import cilisi
from .cluster_batch_entropy import cluster_batch_entropy
from .clustering import cluster_optimal_resolution, kmeans
from .compactness import compactness
from .completeness import completeness
from .corr_mse import corr_mse
from .crossbatch_knn import crossbatch_knn
from .crossbatch_logistic import crossbatch_logistic
from .crossfold_knn import crossfold_knn
from .crossfold_logistic import crossfold_logistic
from .davies_bouldin import davies_bouldin
from .f1_score import f1_score_macro
from .fowlkes_mallows import fowlkes_mallows
from .graph_connectivity import graph_connectivity
from .homogeneity import homogeneity
from .isolated_labels import isolated_labels, isolated_labels_lisi
from .jaccard import jaccard_index
from .kbet import calc_kBET, calc_kBET_across_labels
from .kl_divergence import kl_divergence
from .knn_divergence import knn_divergence
from .ksim import ksim
from .ldf_diff import ldf_diff
from .lisi import clisi_graph, ilisi_graph
from .matthews_corrcoef import matthews_corrcoef
from .mixing import mixing
from .modularity import modularity
from .morans_i import morans_i
from .nmi import nmi, nmi_fixed
from .nnmap import nnmap
from .over_correction import over_correction
from .pcr import pcr_comparison, per_label_pcr_comparison
from .pptpc import pptpc
from .pstdb import pstdb
from .purity import purity, batch_purity
from .retained_knn import retained_knn
from .sc_graph import scgraph
from .scaled_variance import scaled_variance
from .separation import separation
from .shannon_entropy_cluster import batch_shannon_entropy, celltype_shannon_entropy
from .shannon_entropy_nn import shannon_entropy_nn
from .silhouette import silhouette, silhouette_batch, silhouette_cluster
from .trajectory import trajectory_conservation
from .v_measure import v_measure
from .variation_of_information import variation_of_information


def metrics(
    adata,
    adata_int,
    batch_key,
    label_key,
    embed,
    embed_pre="X_pca",
    leiden_cluster_key="leiden_",
    kmeans_cluster_key="kmeans_",
    verbose=False,
    leiden_resolutions=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
    kmeans_resolutions=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
    all_cluster_=False,
    adgbd_=False,
    awc_=False,
    agree_=False,
    align_=False,
    ami_=False,
    ari_=False,
    arif_=False,
    bami_=False,
    bari_=False,
    batchari_=False,
    probe_=False,
    cprobe_=False,
    bp_=False,
    bras_=False,
    bvm_=False,
    clustch_=False,
    ccc_=False,
    cch_=False,
    csms_=False,
    cre_=False,
    cilisi_=False,
    compact_=False,
    comp_=False,
    corrmse_=False,
    bknn_=False,
    blogreg_=False,
    fknn_=False,
    flogreg_=False,
    cdbi_=False,
    clustdbi_=False,
    f1_=False,
    fmi_=False,
    gc_=False,
    homo_=False,
    isof1_=False,
    isoasw_=False,
    isolisi_=False,
    jacc_=False,
    kbet_=False,
    kbetall_=False,
    kldiv_=False,
    embdiv_=False,
    ksim_=False,
    ldfdiff_=False,
    clisi_=False,
    ilisi_=False,
    mcc_=False,
    mix_=False,
    modularity_=False,
    morani_=False,
    nmi_=False,
    nmif_=False,
    nnmap_=False,
    ocs_=False,
    pcr_=False,
    cpcr_=False,
    pptpc_=False,
    pstdb_=False,
    purity_=False,
    rknn_=False,
    scgraph_=False,
    scgraphp_=False,
    scgraphm_=False,
    sv_=False,
    separation_=False,
    bcent_=False,
    ccent_=False,
    bnent_=False,
    cnent_=False,
    casw_=False,
    basw_=False,
    clustasw_=False,
    traj_=False,
    vm_=False,
    vi_=False,
    metric_kwargs=dict(),
):
    """
    Run the specified metrics.

    ``metric_kwargs`` are the keyword arguments for each specific metric. For example,
    ``{'casw_kwargs': {'metric': 'cosine'}}`` would change the CASW's distance metric
    from the default Euclidean to cosine.
    Keys in ``metric_kwargs`` are formatted as ``<name_of_metric>_kwargs``.

    leiden_resolutions - the resolutions for Leiden clustering for clustering-based metrics.
    kmeans_resolutions - the proportion of # cell types to use as k. e.g. 1.0 sets k equal to
        the number of cell types. If the proportion x # cell types is 1, the resolution will be
        ignored.
    """
    check_adata(adata)
    check_batch(batch_key, adata.obs)
    check_batch(label_key, adata.obs)

    check_adata(adata_int)
    check_batch(batch_key, adata_int.obs)
    check_batch(label_key, adata_int.obs)

    if np.any(adata.obs_names != adata_int.obs_names):
        adata_int = adata_int[adata.obs_names].copy()

    if adata_int.X is None:
        adata_int = ad.AnnData(
            X=adata.X,
            obs=adata_int.obs,
            var=adata.var,
            uns=adata_int.uns,
            obsm=adata_int.obsm,
            varm=adata_int.varm,
            obsp=adata_int.obsp,
            varp=adata.varp,
            layers=adata.layers,
        )

    # Get the K-means k's
    kmeans_ks = set()
    n_celltypes = len(np.unique(adata_int.obs[label_key]))
    for r in kmeans_resolutions:
        k = math.ceil(r * n_celltypes)
        if k > 1:
            kmeans_ks.add(k)
    kmeans_ks = sorted(list(kmeans_ks))

    if nmi_ or all_cluster_:
        print("NMI...")
        _, nmi_score_leiden = cluster_optimal_resolution(
            adata_int,
            label_key=label_key,
            cluster_key=leiden_cluster_key,
            metric=nmi,
            resolutions=leiden_resolutions,
        )

        _, nmi_score_kmeans = cluster_optimal_resolution(
            adata_int,
            label_key=label_key,
            cluster_key=kmeans_cluster_key,
            metric=nmi,
            resolutions=kmeans_resolutions,
            cluster_function=kmeans,
        )
        nmi_score = max(nmi_score_leiden, nmi_score_kmeans)
    else:
        nmi_score = np.nan

    if ari_ or all_cluster_:
        print("ARI...")
        _, ari_score_leiden = cluster_optimal_resolution(
            adata_int, label_key=label_key, cluster_key=leiden_cluster_key, metric=ari, resolutions=leiden_resolutions
        )
        _, ari_score_kmeans = cluster_optimal_resolution(
            adata_int, label_key=label_key, cluster_key=kmeans_cluster_key, metric=ari, resolutions=kmeans_resolutions, cluster_function=kmeans
        )
        ari_score = max(ari_score_leiden, ari_score_kmeans)
    else:
        ari_score = np.nan

    if casw_:
        print("Cell type silhouette score...")
        casw_kwargs = convert_values(metric_kwargs.get("casw_kwargs", {}), silhouette)
        asw_label = silhouette(
            adata_int,
            label_key=label_key,
            embed=embed,
            metric=casw_kwargs.get("metric", "euclidean"),
        )
    else:
        asw_label = np.nan

    if basw_:
        print("Batch silhouette score...")
        basw_kwargs = convert_values(
            metric_kwargs.get("basw_kwargs", {}), silhouette_batch
        )
        asw_batch = silhouette_batch(
            adata_int,
            batch_key=batch_key,
            label_key=label_key,
            embed=embed,
            metric=basw_kwargs.get("metric", "euclidean"),
            return_all=False,
            verbose=False,
        )
    else:
        asw_batch = np.nan

    if clustasw_ or all_cluster_:
        print("Cluster ASW score...")
        clustasw_kwargs = convert_values(
            metric_kwargs.get("clustasw_kwargs", {})
            | metric_kwargs.get("all_cluster_kwargs", {}),
            silhouette_cluster,
        )
        _clustasw_metric_kwargs = {
            "embed": embed,
            "metric": clustasw_kwargs.get("metric", "euclidean"),
        }
        _, silhouette_cluster_score_leiden = cluster_optimal_resolution(
            adata_int,
            label_key,
            leiden_cluster_key,
            metric=silhouette_cluster,
            resolutions=leiden_resolutions,
            metric_kwargs=_clustasw_metric_kwargs,
        )
        _, silhouette_cluster_score_kmeans = cluster_optimal_resolution(
            adata_int,
            label_key,
            kmeans_cluster_key,
            metric=silhouette_cluster,
            resolutions=kmeans_resolutions,
            cluster_function=kmeans,
            metric_kwargs=_clustasw_metric_kwargs,
        )
        silhouette_cluster_score = max(silhouette_cluster_score_leiden, silhouette_cluster_score_kmeans)
    else:
        silhouette_cluster_score = np.nan

    if pcr_:
        print("PC regression...")
        pcr_kwargs = convert_values(metric_kwargs.get("pcr_kwargs", {}), pcr_comparison)
        pcr_score = pcr_comparison(
            adata,
            adata_int,
            embed=embed,
            covariate=batch_key,
            verbose=verbose,
            **pcr_kwargs,
        )
    else:
        pcr_score = np.nan

    if cpcr_:
        print("Per cell type PC regression...")
        cpcr_score = per_label_pcr_comparison(
            adata, adata_int, label_key, batch_key, embed=embed
        )
    else:
        cpcr_score = np.nan

    if ccc_:
        print("cell cycle effect...")

        # Small thing to use gene names, not Ensembl IDs
        if "gene_name" in adata.var.columns:
            adata_copy = adata[:, ~adata.var["gene_name"].isna()].copy()
            adata_copy.var = adata_copy.var.set_index("gene_name")
            adata_copy.var_names = adata_copy.var_names.astype("object")
            adata_copy.var_names_make_unique()
        else:
            adata_copy = adata
        if "gene_name" in adata_int.var.columns:
            adata_int_copy = adata_int[:, ~adata_int.var["gene_name"].isna()].copy()
            adata_int_copy.var = adata_int_copy.var.set_index("gene_name")
            adata_int_copy.var_names = adata_int_copy.var_names.astype("object")
            adata_int_copy.var_names_make_unique()
        else:
            adata_int_copy = adata_int

        if "species" not in adata.uns.keys():
            raise ValueError("Expected `species` key in adata.uns")

        cc_score = cell_cycle(
            adata_copy,
            adata_int_copy,
            batch_key=batch_key,
            embed=embed,
            agg_func=np.mean,
            organism=adata.uns["species"],
        )
    else:
        cc_score = np.nan

    if isof1_ or all_cluster_:
        print("Isolated labels F1...")
        il_score_f1 = isolated_labels(
            adata_int,
            label_key=label_key,
            batch_key=batch_key,
            embed=embed,
            cluster=True,
            iso_threshold=metric_kwargs.get("n_isolated", None),
            verbose=False,
        )
    else:
        il_score_f1 = np.nan

    if isoasw_:
        print("Isolated labels ASW...")
        isoasw_kwargs = convert_values(
            metric_kwargs.get("isoasw_kwargs", {}), isolated_labels
        )
        il_score_asw = isolated_labels(
            adata_int,
            label_key=label_key,
            batch_key=batch_key,
            embed=embed,
            cluster=False,
            verbose=False,
            **isoasw_kwargs,
        )
    else:
        il_score_asw = np.nan

    if isolisi_:
        print("Isolated labels CLISI...")
        isolisi_kwargs = metric_kwargs.get("isolisi_kwargs", {})
        if isolisi_kwargs.get("metric", "euclidean") != "euclidean":
            sc.pp.neighbors(adata_int, metric=isolisi_kwargs["metric"], use_rep=embed)
        il_score_lisi = isolated_labels_lisi(
            adata_int, label_key=label_key, batch_key=batch_key, embed=embed
        )
    else:
        il_score_lisi = np.nan

    if gc_:
        print("Graph connectivity...")
        gc_kwargs = convert_values(
            metric_kwargs.get("gc_kwargs", {}), graph_connectivity
        )
        graph_conn_score = graph_connectivity(
            adata_int, label_key=label_key, embed=embed, **gc_kwargs
        )
    else:
        graph_conn_score = np.nan

    if scgraph_:
        print("scGraph...")
        scgraph_kwargs = convert_values(
            metric_kwargs.get("scgraph_kwargs", {}), scgraph
        )
        scgraph_avg, _, _, _ = scgraph(
            adata_int,
            batch_key=batch_key,
            label_key=label_key,
            embed=embed,
            **scgraph_kwargs,
        )
    else:
        scgraph_avg = np.nan

    if scgraphp_:
        print("scGraph with Fisher Poisson distance...")
        scgraphp_kwargs = convert_values(
            metric_kwargs.get("scgraphp_kwargs", {}), scgraph
        )
        scgraphp_avg, _, _, _ = scgraph(
            adata_int,
            batch_key=batch_key,
            label_key=label_key,
            embed=embed,
            method="fisher_poisson",
            **scgraphp_kwargs,
        )
    else:
        scgraphp_avg = np.nan

    if scgraphm_:
        print("scGraph with Fisher multinomial distance...")
        scgraphm_kwargs = convert_values(
            metric_kwargs.get("scgraphm_kwargs", {}), scgraph
        )
        scgraphm_avg, _, _, _ = scgraph(
            adata_int,
            batch_key=batch_key,
            label_key=label_key,
            embed=embed,
            method="fisher_multinomial",
            **scgraphm_kwargs,
        )
    else:
        scgraphm_avg = np.nan

    if kbet_:
        print("kBET...")
        kbet_kwargs = convert_values(
            metric_kwargs.get("kbet_kwargs", {}), calc_kBET_across_labels
        )
        _, _, kbet_score = calc_kBET_across_labels(
            adata_int, batch_key, label_key, embed, **kbet_kwargs
        )
    else:
        kbet_score = np.nan

    if kbetall_:
        kbet_kwargs = convert_values(metric_kwargs.get("kbetall_kwargs", {}), calc_kBET)
        _, _, kbet_no_label_score = calc_kBET(
            adata_int, batch_key, embed, **kbet_kwargs
        )
    else:
        kbet_no_label_score = np.nan

    if clisi_:
        print("cLISI score...")
        clisi_kwargs = convert_values(
            metric_kwargs.get("clisi_kwargs", {}), clisi_graph
        )
        clisi_score = clisi_graph(
            adata_int,
            batch_key=batch_key,
            label_key=label_key,
            type_="knn",
            use_rep=embed,
            scale=True,
            verbose=verbose,
            **clisi_kwargs,
        )
    else:
        clisi_score = np.nan

    if ilisi_:
        print("iLISI score...")
        ilisi_kwargs = convert_values(
            metric_kwargs.get("ilisi_kwargs", {}), ilisi_graph
        )
        ilisi_score = ilisi_graph(
            adata_int,
            batch_key=batch_key,
            type_="knn",
            scale=True,
            use_rep=embed,
            verbose=verbose,
            **ilisi_kwargs,
        )
    else:
        ilisi_score = np.nan

    if cilisi_:
        print("CiLISI score...")
        cilisi_kwargs = convert_values(metric_kwargs.get("cilisi_kwargs", {}), cilisi)
        cilisi_score = cilisi(adata_int, batch_key, label_key, embed, **cilisi_kwargs)
    else:
        cilisi_score = np.nan

    if traj_:
        print("Trajectory conservation score...")
        trajectory_score = trajectory_conservation(
            adata,
            adata_int,
            label_key=label_key,
        )
    else:
        trajectory_score = np.nan

    if ldfdiff_:
        print("LDF difference score...")
        ldfdiff_kwargs = convert_values(
            metric_kwargs.get("ldfdiff_kwargs", {}), ldf_diff
        )
        ldf_diff_score = ldf_diff(
            adata_pre=adata,
            adata_post=adata_int,
            batch_key=batch_key,
            embed_pre=embed_pre,
            embed_post=embed,
            **ldfdiff_kwargs,
        )
    else:
        ldf_diff_score = np.nan

    if csms_:
        print("Cell-specific mixing score...")
        csms_kwargs = convert_values(
            metric_kwargs.get("csms_kwargs", {}), cell_specific_mixing_score
        )
        cell_specific_mixing_score_result = cell_specific_mixing_score(
            adata_int, batch_key=batch_key, embed=embed, **csms_kwargs
        )
    else:
        cell_specific_mixing_score_result = np.nan

    if align_:
        print("Alignment score...")
        align_kwargs = convert_values(metric_kwargs.get("align_kwargs", {}), alignment)
        alignment_score = alignment(adata_int, batch_key, embed, **align_kwargs)
    else:
        alignment_score = np.nan

    if agree_:
        print("Agreement score...")
        agree_kwargs = convert_values(metric_kwargs.get("agree_kwargs", {}), agreement)
        agreement_score = agreement(
            adata, adata_int, batch_key, label_key, embed, **agree_kwargs
        )
    else:
        agreement_score = np.nan

    if bknn_:
        print("Cross-batch k-NN score...")
        bknn_kwargs = convert_values(
            metric_kwargs.get("bknn_kwargs", {}), crossbatch_knn
        )
        crossbatch_knn_score = crossbatch_knn(
            adata_int, batch_key, label_key, embed, **bknn_kwargs
        )
    else:
        crossbatch_knn_score = np.nan

    if blogreg_:
        print("Cross-batch logistic regression...")
        crossbatch_logistic_score = crossbatch_logistic(
            adata_int,
            batch_key,
            label_key,
            embed,
        )
    else:
        crossbatch_logistic_score = np.nan

    if fknn_:
        print("Cross-fold k-NN score...")
        fknn_kwargs = convert_values(
            metric_kwargs.get("fknn_kwargs", {}), crossfold_knn
        )
        crossfold_knn_score = crossfold_knn(adata_int, label_key, embed, **fknn_kwargs)
    else:
        crossfold_knn_score = np.nan

    if flogreg_:
        print("Cross-fold logistic regression...")
        crossfold_logistic_score = crossfold_logistic(adata_int, label_key, embed)
    else:
        crossfold_logistic_score = np.nan

    if batchari_ or all_cluster_:
        print("Batch ARI...")
        _, batch_ari_score_leiden = cluster_optimal_resolution(
            adata_int,
            label_key=batch_key,
            cluster_key=leiden_cluster_key,
            metric=batch_ari,
            resolutions=leiden_resolutions,
        )
        _, batch_ari_score_kmeans = cluster_optimal_resolution(
            adata_int,
            label_key=batch_key,
            cluster_key=kmeans_cluster_key,
            metric=batch_ari,
            resolutions=kmeans_resolutions,
            cluster_function=kmeans,
        )
        batch_ari_score = max(batch_ari_score_leiden, batch_ari_score_kmeans)
    else:
        batch_ari_score = np.nan

    if bp_ or all_cluster_:
        print("Batch purity score...")
        _, batch_purity_score_leiden = cluster_optimal_resolution(
            adata_int,
            label_key=batch_key,
            cluster_key=leiden_cluster_key,
            metric=batch_purity,
            resolutions=leiden_resolutions,
        )
        _, batch_purity_score_kmeans = cluster_optimal_resolution(
            adata_int,
            label_key=batch_key,
            cluster_key=kmeans_cluster_key,
            metric=batch_purity,
            resolutions=kmeans_resolutions,
            cluster_function=kmeans,
        )
        batch_purity_score = max(batch_purity_score_leiden, batch_purity_score_kmeans)
    else:
        batch_purity_score = np.nan

    if bcent_ or all_cluster_:
        print("Batch Shannon entropy by cluster...")
        _, batch_shannon_entropy_score_leiden = cluster_optimal_resolution(
            adata_int,
            label_key=batch_key,
            cluster_key=leiden_cluster_key,
            metric=batch_shannon_entropy,
            resolutions=leiden_resolutions,
        )
        _, batch_shannon_entropy_score_kmeans = cluster_optimal_resolution(
            adata_int,
            label_key=batch_key,
            cluster_key=kmeans_cluster_key,
            metric=batch_shannon_entropy,
            resolutions=kmeans_resolutions,
            cluster_function=kmeans,
        )
        batch_shannon_entropy_score = max(batch_shannon_entropy_score_leiden, batch_shannon_entropy_score_kmeans)
    else:
        batch_shannon_entropy_score = np.nan

    if ccent_ or all_cluster_:
        print("Cell type Shannon entropy by cluster...")
        _, celltype_shannon_entropy_score_leiden = cluster_optimal_resolution(
            adata_int, label_key, leiden_cluster_key, metric=celltype_shannon_entropy, resolutions=leiden_resolutions
        )
        _, celltype_shannon_entropy_score_kmeans = cluster_optimal_resolution(
            adata_int, label_key, kmeans_cluster_key, metric=celltype_shannon_entropy, resolutions=kmeans_resolutions, cluster_function=kmeans
        )
        celltype_shannon_entropy_score = max(celltype_shannon_entropy_score_leiden, celltype_shannon_entropy_score_kmeans)
    else:
        celltype_shannon_entropy_score = np.nan

    if bnent_ and cnent_:
        print("Shannon entropy nearest neighbor graph...")
        entropy_kwargs = convert_values(
            metric_kwargs.get("bnent_kwargs", {}), shannon_entropy_nn
        )
        batch_shannon_entropy_nn_score, cell_shannon_entropy_nn_score = (
            shannon_entropy_nn(
                adata_int, batch_key, label_key, embed=embed, **entropy_kwargs
            )
        )
    elif bnent_:
        print("Shannon entropy nearest neighbor graph...")
        entropy_kwargs = convert_values(
            metric_kwargs.get("bnent_kwargs", {}), shannon_entropy_nn
        )
        batch_shannon_entropy_nn_score, _ = shannon_entropy_nn(
            adata_int, batch_key, label_key, embed=embed, **entropy_kwargs
        )
        cell_shannon_entropy_nn_score = np.nan
    elif cnent_:
        print("Shannon entropy nearest neighbor graph...")
        entropy_kwargs = convert_values(
            metric_kwargs.get("cnent_kwargs", {}), shannon_entropy_nn
        )
        _, cell_shannon_entropy_nn_score = shannon_entropy_nn(
            adata_int, batch_key, label_key, embed=embed, **entropy_kwargs
        )
        batch_shannon_entropy_nn_score = np.nan
    else:
        batch_shannon_entropy_nn_score = cell_shannon_entropy_nn_score = np.nan

    if arif_:
        print("ARI fixed...")
        arif_kwargs = metric_kwargs.get("arif_kwargs", {})
        if arif_kwargs.get("metric", "euclidean") != "euclidean":
            sc.pp.neighbors(adata_int, metric=arif_kwargs["metric"], use_rep=embed)
        arif_score = ari_fixed(adata_int, label_key, leiden_cluster_key, embed=embed)
    else:
        arif_score = np.nan

    if bari_ or all_cluster_:
        print("Balanced ARI...")
        _, bARI_score_leiden = cluster_optimal_resolution(
            adata_int, label_key=label_key, cluster_key=leiden_cluster_key, metric=bARI, resolutions=leiden_resolutions
        )
        _, bARI_score_kmeans = cluster_optimal_resolution(
            adata_int, label_key=label_key, cluster_key=kmeans_cluster_key, metric=bARI, resolutions=kmeans_resolutions, cluster_function=kmeans
        )
        bARI_score = max(bARI_score_leiden, bARI_score_kmeans)
    else:
        bARI_score = np.nan

    if bami_ or all_cluster_:
        print("Balanced AMI...")
        _, bami_score_leiden = cluster_optimal_resolution(
            adata_int, label_key=label_key, cluster_key=leiden_cluster_key, metric=bAMI, resolutions=leiden_resolutions
        )
        _, bami_score_kmeans = cluster_optimal_resolution(
            adata_int, label_key=label_key, cluster_key=kmeans_cluster_key, metric=bAMI, resolutions=kmeans_resolutions, cluster_function=kmeans
        )
        bami_score = max(bami_score_leiden, bami_score_kmeans)
    else:
        bami_score = np.nan

    if bvm_ or all_cluster_:
        print("Balanced v-measure...")
        _, bv_measure_score_leiden = cluster_optimal_resolution(
            adata_int,
            label_key=label_key,
            cluster_key=leiden_cluster_key,
            metric=bv_measure,
            resolutions=leiden_resolutions,
        )
        _, bv_measure_score_kmeans = cluster_optimal_resolution(
            adata_int,
            label_key=label_key,
            cluster_key=kmeans_cluster_key,
            metric=bv_measure,
            resolutions=kmeans_resolutions,
            cluster_function=kmeans,
        )
        bv_measure_score = max(bv_measure_score_leiden, bv_measure_score_kmeans)
    else:
        bv_measure_score = np.nan

    if probe_:
        print("Batch probing score...")
        bps_rf, bps_lr = batch_probing_score(adata_int, batch_key, embed)
    else:
        bps_rf = bps_lr = np.nan

    if cprobe_:
        print("Per cell type batch probing score...")
        cprobe_rf, cprobe_lr = per_label_batch_probing_score(
            adata_int, label_key, batch_key, embed
        )
    else:
        cprobe_rf = cprobe_lr = np.nan

    if clustch_ or all_cluster_:
        print("Calinkski Harabasz score...")
        _, ch_score_leiden = cluster_optimal_resolution(
            adata_int,
            label_key=label_key,
            cluster_key=leiden_cluster_key,
            metric=calinski_harabasz,
            resolutions=leiden_resolutions,
            metric_kwargs={"embed": embed},
        )
        _, ch_score_kmeans = cluster_optimal_resolution(
            adata_int,
            label_key=label_key,
            cluster_key=kmeans_cluster_key,
            metric=calinski_harabasz,
            resolutions=kmeans_resolutions,
            cluster_function=kmeans,
            metric_kwargs={"embed": embed},
        )
        ch_score = max(ch_score_leiden, ch_score_kmeans)
    else:
        ch_score = np.nan

    if cch_:
        print("Cell type Calinski Harabasz...")
        cch_score = calinski_harabasz(adata_int, None, label_key, embed)
    else:
        cch_score = np.nan

    if f1_ or all_cluster_:
        print("F1 score...")
        _, f1sm_leiden = cluster_optimal_resolution(
            adata_int,
            label_key=label_key,
            cluster_key=leiden_cluster_key,
            metric=f1_score_macro,
            resolutions=leiden_resolutions,
        )
        _, f1sm_kmeans = cluster_optimal_resolution(
            adata_int,
            label_key=label_key,
            cluster_key=kmeans_cluster_key,
            metric=f1_score_macro,
            resolutions=kmeans_resolutions,
            cluster_function=kmeans,
        )
        f1sm = max(f1sm_leiden, f1sm_kmeans)
    else:
        f1sm = np.nan

    if compact_ or all_cluster_:
        print("Compactness...")
        _, compactness_score_leiden = cluster_optimal_resolution(
            adata_int,
            label_key=label_key,
            cluster_key=leiden_cluster_key,
            metric=compactness,
            resolutions=leiden_resolutions,
            metric_kwargs={"embed": embed},
        )
        _, compactness_score_kmeans = cluster_optimal_resolution(
            adata_int,
            label_key=label_key,
            cluster_key=kmeans_cluster_key,
            metric=compactness,
            resolutions=kmeans_resolutions,
            cluster_function=kmeans,
            metric_kwargs={"embed": embed},
        )
        compactness_score = max(compactness_score_leiden, compactness_score_kmeans)
    else:
        compactness_score = np.nan

    if corrmse_:
        print("Corr-MSE...")
        corrmse_score = corr_mse(adata, adata_int, batch_key, embed, embed_pre)
    else:
        corrmse_score = np.nan

    if kldiv_:
        print("KL divergence...")
        kldiv_kwargs = convert_values(
            metric_kwargs.get("kldiv_kwargs", {}), kl_divergence
        )
        kl_div_score = kl_divergence(adata_int, batch_key, embed, **kldiv_kwargs)
    else:
        kl_div_score = np.nan

    if homo_ or all_cluster_:
        print("Homogeneity...")
        _, homogeneity_score_leiden = cluster_optimal_resolution(
            adata_int,
            label_key,
            leiden_cluster_key,
            metric=homogeneity,
            resolutions=leiden_resolutions,
        )
        _, homogeneity_score_kmeans = cluster_optimal_resolution(
            adata_int,
            label_key,
            kmeans_cluster_key,
            metric=homogeneity,
            resolutions=kmeans_resolutions,
            cluster_function=kmeans,
        )
        homogeneity_score = max(homogeneity_score_leiden, homogeneity_score_kmeans)
    else:
        homogeneity_score = np.nan

    if fmi_ or all_cluster_:
        print("Fowlkes Mallows...")
        _, fm_score_leiden = cluster_optimal_resolution(
            adata_int, label_key, leiden_cluster_key, metric=fowlkes_mallows, resolutions=leiden_resolutions
        )
        _, fm_score_kmeans = cluster_optimal_resolution(
            adata_int, label_key, kmeans_cluster_key, metric=fowlkes_mallows, resolutions=kmeans_resolutions, cluster_function=kmeans
        )
        fm_score = max(fm_score_leiden, fm_score_kmeans)
    else:
        fm_score = np.nan

    if jacc_ or all_cluster_:
        print("Jaccard index...")
        _, jaccard_score_leiden = cluster_optimal_resolution(
            adata_int, label_key, leiden_cluster_key, metric=jaccard_index, resolutions=leiden_resolutions
        )
        _, jaccard_score_kmeans = cluster_optimal_resolution(
            adata_int, label_key, kmeans_cluster_key, metric=jaccard_index, resolutions=kmeans_resolutions, cluster_function=kmeans
        )
        jaccard_score = max(jaccard_score_leiden, jaccard_score_kmeans)
    else:
        jaccard_score = np.nan

    if mcc_ or all_cluster_:
        print("Matthews correlation coefficient")
        _, mcc_score_leiden = cluster_optimal_resolution(
            adata_int, label_key, leiden_cluster_key, metric=matthews_corrcoef, resolutions=leiden_resolutions
        )
        _, mcc_score_kmeans = cluster_optimal_resolution(
            adata_int, label_key, kmeans_cluster_key, metric=matthews_corrcoef, resolutions=kmeans_resolutions, cluster_function=kmeans
        )
        mcc_score = max(mcc_score_leiden, mcc_score_kmeans)
    else:
        mcc_score = np.nan

    if modularity_ or all_cluster_:
        print("Modularity...")
        _, modularity_score_leiden = cluster_optimal_resolution(
            adata_int, label_key, leiden_cluster_key, metric=modularity, resolutions=leiden_resolutions
        )
        _, modularity_score_kmeans = cluster_optimal_resolution(
            adata_int, label_key, kmeans_cluster_key, metric=modularity, resolutions=kmeans_resolutions, cluster_function=kmeans
        )
        modularity_score = max(modularity_score_leiden, modularity_score_kmeans)
    else:
        modularity_score = np.nan

    if adgbd_:
        print("Average deviation from global batch distribution...")
        adgbd_kwargs = convert_values(metric_kwargs.get("adgbd_kwargs", {}), adgbd)
        adgbd_score = adgbd(adata_int, batch_key, embed=embed, **adgbd_kwargs)
    else:
        adgbd_score = np.nan

    if awc_ or all_cluster_:
        print("Adjusted Wallace score...")
        _, aw_score_leiden = cluster_optimal_resolution(
            adata_int, label_key, leiden_cluster_key, metric=adjusted_wallace_score, resolutions=leiden_resolutions
        )
        _, aw_score_kmeans = cluster_optimal_resolution(
            adata_int, label_key, kmeans_cluster_key, metric=adjusted_wallace_score, resolutions=kmeans_resolutions, cluster_function=kmeans
        )
        aw_score = max(aw_score_leiden, aw_score_kmeans)
    else:
        aw_score = np.nan

    if ami_ or all_cluster_:
        print("Adjusted mutual information")
        _, ami_score_leiden = cluster_optimal_resolution(
            adata_int, label_key, leiden_cluster_key, metric=ami, resolutions=leiden_resolutions
        )
        _, ami_score_kmeans = cluster_optimal_resolution(
            adata_int, label_key, kmeans_cluster_key, metric=ami, resolutions=kmeans_resolutions, cluster_function=kmeans
        )
        ami_score = max(ami_score_leiden, ami_score_kmeans)
    else:
        ami_score = np.nan

    if bras_:
        print("BRAS...")
        bras_kwargs = convert_values(metric_kwargs.get("bras_kwargs", {}), bras)
        bras_score = bras(adata_int, label_key, batch_key, embed, **bras_kwargs)
    else:
        bras_score = np.nan

    if cre_ or all_cluster_:
        print("Cluster batch entropy...")
        _, cluster_batch_entropy_score_leiden = cluster_optimal_resolution(
            adata_int, batch_key, leiden_cluster_key, metric=cluster_batch_entropy, resolutions=leiden_resolutions
        )
        _, cluster_batch_entropy_score_kmeans = cluster_optimal_resolution(
            adata_int, batch_key, kmeans_cluster_key, metric=cluster_batch_entropy, resolutions=kmeans_resolutions, cluster_function=kmeans
        )
        cluster_batch_entropy_score = max(cluster_batch_entropy_score_leiden, cluster_batch_entropy_score_kmeans)
    else:
        cluster_batch_entropy_score = np.nan

    if comp_ or all_cluster_:
        print("Completeness...")
        _, completeness_score_leiden = cluster_optimal_resolution(
            adata_int, label_key, leiden_cluster_key, metric=completeness, resolutions=leiden_resolutions
        )
        _, completeness_score_kmeans = cluster_optimal_resolution(
            adata_int, label_key, kmeans_cluster_key, metric=completeness, resolutions=kmeans_resolutions, cluster_function=kmeans
        )
        completeness_score = max(completeness_score_leiden, completeness_score_kmeans)
    else:
        completeness_score = np.nan

    if cdbi_:
        print("Cell type Davies-Bouldin...")
        cdbi_score = davies_bouldin(adata_int, None, label_key, embed)
    else:
        cdbi_score = np.nan

    if clustdbi_ or all_cluster_:
        print("Cluster Davies-Bouldin...")
        _, clust_dbi_score_leiden = cluster_optimal_resolution(
            adata_int,
            label_key,
            leiden_cluster_key,
            metric=davies_bouldin,
            resolutions=leiden_resolutions,
            metric_kwargs={"embed": embed},
        )
        _, clust_dbi_score_kmeans = cluster_optimal_resolution(
            adata_int,
            label_key,
            kmeans_cluster_key,
            metric=davies_bouldin,
            resolutions=kmeans_resolutions,
            cluster_function=kmeans,
            metric_kwargs={"embed": embed},
        )
        clust_dbi_score = max(clust_dbi_score_leiden, clust_dbi_score_kmeans)
    else:
        clust_dbi_score = np.nan

    if embdiv_:
        print("KNN divergence...")
        knn_div_score = knn_divergence(adata_int, label_key, batch_key, embed)
    else:
        knn_div_score = np.nan

    if ksim_:
        print("kSIM...")
        ksim_kwargs = convert_values(metric_kwargs.get("ksim_kwargs", {}), ksim)
        ksim_score = ksim(adata_int, label_key, embed, **ksim_kwargs)
    else:
        ksim_score = np.nan

    if mix_:
        print("Mixing score...")
        mix_kwargs = convert_values(metric_kwargs.get("mix_kwargs", {}), mixing)
        mixing_score = mixing(adata_int, batch_key, embed, **mix_kwargs)
    else:
        mixing_score = np.nan

    if morani_:
        print("Moran's I...")
        moransi_score = morans_i(adata, adata_int, label_key, batch_key, "X_pca", embed)
    else:
        moransi_score = np.nan

    if nmif_:
        print("NMI fixed...")
        nmif_kwargs = metric_kwargs.get("nmif_kwargs", {})
        if nmif_kwargs.get("metric", "euclidean") != "euclidean":
            sc.pp.neighbors(adata_int, metric=nmif_kwargs["metric"], use_rep=embed)
        nmi_fixed_score = nmi_fixed(adata_int, label_key, leiden_cluster_key)
    else:
        nmi_fixed_score = np.nan

    if nnmap_:
        print("Nearest neighbor mean average precision")
        nnmap_kwargs = convert_values(metric_kwargs.get("nnmap_kwargs", {}), nnmap)
        nnmap_score = nnmap(adata_int, label_key, embed, **nnmap_kwargs)
    else:
        nnmap_score = np.nan

    if ocs_:
        print("Over correction...")
        ocs_kwargs = convert_values(
            metric_kwargs.get("ocs_kwargs", {}), over_correction
        )
        oc_score = over_correction(adata_int, label_key, embed=embed, **ocs_kwargs)
    else:
        oc_score = np.nan

    if pptpc_:
        print("Positive of positive and true positive cells...")
        pptpc_kwargs = convert_values(metric_kwargs.get("pptpc_kwargs", {}), pptpc)
        pptpc_score = pptpc(
            adata_int, batch_key, label_key, embed=embed, **pptpc_kwargs
        )
    else:
        pptpc_score = np.nan

    if pstdb_:
        print("Proportion of same cell type but different batch...")
        pstdb_kwargs = convert_values(metric_kwargs.get("pstdb_kwargs", {}), pstdb)
        pstdb_score = pstdb(adata_int, batch_key, label_key, embed, **pstdb_kwargs)
    else:
        pstdb_score = np.nan

    if purity_ or all_cluster_:
        print("Purity...")
        _, purity_score_leiden = cluster_optimal_resolution(
            adata_int, label_key, leiden_cluster_key, metric=purity, resolutions=leiden_resolutions
        )
        _, purity_score_kmeans = cluster_optimal_resolution(
            adata_int, label_key, kmeans_cluster_key, metric=purity, resolutions=kmeans_resolutions, cluster_function=kmeans
        )
        purity_score = max(purity_score_leiden, purity_score_kmeans)
    else:
        purity_score = np.nan

    if rknn_:
        print("Retained KNN...")
        rknn_kwargs = convert_values(metric_kwargs.get("rknn_kwargs", {}), retained_knn)
        retained_knn_score = retained_knn(
            adata, adata_int, batch_key, embed, **rknn_kwargs
        )
    else:
        retained_knn_score = np.nan

    if sv_:
        print("Scaled variance...")
        sv_score = scaled_variance(adata_int, batch_key, embed)
    else:
        sv_score = np.nan

    if separation_ or all_cluster_:
        _, separation_score_leiden = cluster_optimal_resolution(
            adata_int,
            label_key,
            cluster_key=leiden_cluster_key,
            metric=separation,
            resolutions=leiden_resolutions,
            metric_kwargs={"embed": embed},
        )
        _, separation_score_kmeans = cluster_optimal_resolution(
            adata_int,
            label_key,
            cluster_key=kmeans_cluster_key,
            metric=separation,
            resolutions=kmeans_resolutions,
            cluster_function=kmeans,
            metric_kwargs={"embed": embed},
        )
        separation_score = max(separation_score_leiden, separation_score_kmeans)
    else:
        separation_score = np.nan

    if vm_ or all_cluster_:
        print("V measure...")
        _, v_score_leiden = cluster_optimal_resolution(
            adata_int, label_key, cluster_key=leiden_cluster_key, metric=v_measure, resolutions=leiden_resolutions
        )
        _, v_score_kmeans = cluster_optimal_resolution(
            adata_int, label_key, cluster_key=kmeans_cluster_key, metric=v_measure, resolutions=kmeans_resolutions, cluster_function=kmeans
        )
        v_score = max(v_score_leiden, v_score_kmeans)
    else:
        v_score = np.nan

    if vi_ or all_cluster_:
        print("Variation of information...")
        _, vi_score_leiden = cluster_optimal_resolution(
            adata_int,
            label_key,
            cluster_key=leiden_cluster_key,
            metric=variation_of_information,
            resolutions=leiden_resolutions,
        )
        _, vi_score_kmeans = cluster_optimal_resolution(
            adata_int,
            label_key,
            cluster_key=kmeans_cluster_key,
            metric=variation_of_information,
            resolutions=kmeans_resolutions,
            cluster_function=kmeans,
        )
        vi_score = max(vi_score_leiden, vi_score_kmeans)
    else:
        vi_score = np.nan

    results = {
        "ADGBD": adgbd_score,
        "AWC": aw_score,
        "ALIGN": alignment_score,
        "AGREE": agreement_score,
        "AMI": ami_score,
        "ARI": ari_score,
        "ARIF": arif_score,
        "BASW": asw_batch,
        "BAMI": bami_score,
        "BVM": bv_measure_score,
        "BARI": bARI_score,
        "PROBE": (bps_rf + bps_lr) / 2,
        "PROBERF": bps_rf,
        "PROBELR": bps_lr,
        "CPROBE": (cprobe_rf + cprobe_lr) / 2,
        "CPROBERF": cprobe_rf,
        "CPROBELR": cprobe_lr,
        "BATCHARI": batch_ari_score,
        "BP": batch_purity_score,
        "BCENT": batch_shannon_entropy_score,
        "BRAS": bras_score,
        "CCC": cc_score,
        "CCH": cch_score,
        "CSMS": cell_specific_mixing_score_result,
        "CCENT": celltype_shannon_entropy_score,
        "CLISI": clisi_score,
        "CILISI": cilisi_score,
        "CLUSTASW": silhouette_cluster_score,
        "CLUSTDBI": clust_dbi_score,
        "CLUSTCH": ch_score,
        "CRE": cluster_batch_entropy_score,
        "COMPACT": compactness_score,
        "COMP": completeness_score,
        "CORRMSE": corrmse_score,
        "BKNN": crossbatch_knn_score,
        "BLOGREG": crossbatch_logistic_score,
        "FKNN": crossfold_knn_score,
        "FLOGREG": crossfold_logistic_score,
        "CDBI": cdbi_score,
        "F1": f1sm,
        "FMI": fm_score,
        "GC": graph_conn_score,
        "HOMO": homogeneity_score,
        "ILISI": ilisi_score,
        "ISOF1": il_score_f1,
        "ISOASW": il_score_asw,
        "ISOLISI": il_score_lisi,
        "JACC": jaccard_score,
        "KBET": kbet_score,
        "KBETALL": kbet_no_label_score,
        "KLDIV": kl_div_score,
        "EMBDIV": knn_div_score,
        "KSIM": ksim_score,
        "LDFDIFF": ldf_diff_score,
        "MCC": mcc_score,
        "MIX": mixing_score,
        "MODULARITY": modularity_score,
        "MORANI": moransi_score,
        "NMI": nmi_score,
        "NMIF": nmi_fixed_score,
        "NNMAP": nnmap_score,
        "OCS": oc_score,
        "PCR": pcr_score,
        "CPCR": cpcr_score,
        "PPTPC": pptpc_score,
        "PSTDB": pstdb_score,
        "PURITY": purity_score,
        "RKNN": retained_knn_score,
        "SV": sv_score,
        "SCGRAPH": scgraph_avg,
        "SCGRAPHP": scgraphp_avg,
        "SCGRAPHM": scgraphm_avg,
        "SEPARATION": separation_score,
        "BNENT": batch_shannon_entropy_nn_score,
        "CNENT": cell_shannon_entropy_nn_score,
        "CASW": asw_label,
        "TRAJ": trajectory_score,
        "VM": v_score,
        "VI": vi_score,
    }

    return pd.DataFrame.from_dict(results, orient="index")


def convert_values(kwargs, function):
    sig = inspect.signature(function)
    parameters = sig.parameters

    converted_kwargs = {}
    for key, value in kwargs.items():
        if key in parameters:
            target_type = parameters[key].annotation

            # Only convert if a specific type (like int, float, bool) is hinted
            if isinstance(target_type, type) and isinstance(value, str):
                try:
                    # Special handling for booleans since bool("False") is True
                    if target_type is bool:
                        converted_kwargs[key] = value.lower() == "true"
                    else:
                        converted_kwargs[key] = target_type(value)
                except ValueError:
                    # Fallback to original value if casting fails
                    converted_kwargs[key] = value
            else:
                converted_kwargs[key] = value
        else:
            # If the key isn't in the signature (e.g. passed to **kwargs inside), keep as is
            converted_kwargs[key] = value
    return converted_kwargs

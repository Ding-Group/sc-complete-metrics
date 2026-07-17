import inspect

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
from .clustering import cluster_optimal_resolution
from .clustering_accuracy import clustering_accuracy
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
from .highly_variable_genes import hvg_overlap
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
from .shannon_entropy_nn import shannon_entropy_nn, shannon_entropy_nn_weighted
from .silhouette import silhouette, silhouette_batch, silhouette_cluster
from .trajectory import trajectory_conservation
from .v_measure import v_measure
from .variation_of_information import variation_of_information


def metrics_fast(adata, adata_int, batch_key, label_key, **kwargs):
    return metrics(
        adata,
        adata_int,
        batch_key,
        label_key,
        isolated_labels_asw_=True,
        silhouette_=True,
        hvg_=True,
        gc_=True,
        pcr_=True,
        **kwargs,
    )


def metrics_slim(adata, adata_int, batch_key, label_key, **kwargs):
    return metrics(
        adata,
        adata_int,
        batch_key,
        label_key,
        isolated_labels_asw_=True,
        silhouette_=True,
        hvg_=True,
        gc_=True,
        pcr_=True,
        isolated_labels_f1_=True,
        traj_=True,
        nmi_=True,
        ari_=True,
        ccc_=True,
        sc_graph_=True,
        **kwargs,
    )


def metrics_all(adata, adata_int, batch_key, label_key, **kwargs):
    return metrics(
        adata,
        adata_int,
        batch_key,
        label_key,
        isolated_labels_asw_=True,
        silhouette_=True,
        hvg_=True,
        gc_=True,
        pcr_=True,
        isolated_labels_f1_=True,
        traj_=True,
        nmi_=True,
        ari_=True,
        ccc_=True,
        kbet_=True,
        ilisi_=True,
        clisi_=True,
        sc_graph_=True,
        ldfdiff_=True,
        csmc_=True,
        bknn_=True,
        blogreg_=True,
        batchari_=True,
        bps_=True,
        bcentropy_=True,
        **kwargs,
    )


def metrics(
    adata,
    adata_int,
    batch_key,
    label_key,
    embed,
    embed_pre="X_pca",
    cluster_key="cluster",
    organism="human",
    verbose=False,
    knn_only=False,  # No embed available, only KNN graph
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
    bps_=False,
    bras_=False,
    bvm_=False,
    ch_=False,
    ccc_=False,
    cch_=False,
    csmc_=False,
    cbe_=False,
    cilisi_=False,
    clustacc_=False,
    compact_=False,
    comp_=False,
    corrmse_=False,
    bknn_=False,
    blogreg_=False,
    fknn_=False,
    flogreg_=False,
    dbi_=False,
    clustdbi_=False,
    f1_=False,
    fmi_=False,
    gc_=False,
    hvg_=False,
    homo_=False,
    isof1_=False,
    isoasw_=False,
    isolisi_=False,
    jacc_=False,
    kbet_=False,
    kbetall_=False,
    kldiv_=False,
    knndiv_=False,
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
    bcentropy_=False,
    ccentropy_=False,
    bnentropy_=False,
    cnentropy_=False,
    bnentropyw_=False,
    cnentropyw_=False,
    casw_=False,
    basw_=False,
    clustasw_=False,
    traj_=False,
    vm_=False,
    vi_=False,
    metric_kwargs=dict(),
):
    """
    Run the metrics specified.

    metric_kwargs are the keyword arguments for each specific metric.
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

    force = False
    if all_cluster_ and metric_kwargs.get('all_cluster_kwargs'):
        # Recalculate the nearest neighbor graph using the metric
        # if the distance metric is not Euclidean
        params = metric_kwargs['all_cluster_kwargs']
        if params.get('metric', 'euclidean') != 'euclidean':
            sc.pp.neighbors(adata_int, metric=params['metric'], use_rep=embed)
            force = True

    if nmi_ or all_cluster_:
        print("NMI...")
        _, nmi_score = cluster_optimal_resolution(
            adata_int, label_key=label_key, cluster_key=cluster_key, metric=nmi, force=force
        )
    else:
        nmi_score = np.nan

    if ari_ or all_cluster_:
        print("ARI...")
        _, ari_score = cluster_optimal_resolution(
            adata_int, label_key=label_key, cluster_key=cluster_key, metric=ari
        )
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
        basw_kwargs = convert_values(metric_kwargs.get("basw_kwargs", {}), silhouette_batch)
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
        clustasw_kwargs = convert_values(metric_kwargs.get("clustasw_kwargs", {}) | metric_kwargs.get("all_cluster_kwargs", {}), silhouette_cluster)
        _, silhouette_cluster_score = cluster_optimal_resolution(
            adata_int,
            label_key,
            cluster_key,
            metric=silhouette_cluster,
            metric_kwargs={"embed": embed, "metric": clustasw_kwargs.get('metric', 'euclidean')},
        )
    else:
        silhouette_cluster_score = np.nan

    if pcr_:
        print("PC regression...")
        pcr_kwargs = convert_values(metric_kwargs.get("pcr_kwargs", {}), pcr_comparison)
        pcr_score = pcr_comparison(
            adata, adata_int, embed=embed, covariate=batch_key, verbose=verbose, **pcr_kwargs
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
        isoasw_kwargs = convert_values(metric_kwargs.get("isoasw_kwargs", {}), isolated_labels)
        il_score_asw = isolated_labels(
            adata_int,
            label_key=label_key,
            batch_key=batch_key,
            embed=embed,
            cluster=False,
            verbose=False,
            **isoasw_kwargs
        )
    else:
        il_score_asw = np.nan

    if isolisi_:
        print("Isolated labels CLISI...")
        isolisi_kwargs = metric_kwargs.get("isolisi_kwargs", {})
        if isolisi_kwargs.get('metric', 'euclidean') != 'euclidean':
            sc.pp.neighbors(adata_int, metric=isolisi_kwargs['metric'], use_rep=embed)
        il_score_lisi = isolated_labels_lisi(
            adata_int,
            label_key=label_key,
            batch_key=batch_key,
            embed=embed
        )
    else:
        il_score_lisi = np.nan

    if gc_:
        print("Graph connectivity...")
        gc_kwargs = convert_values(metric_kwargs.get("gc_kwargs", {}), graph_connectivity)
        graph_conn_score = graph_connectivity(adata_int, label_key=label_key, embed=embed, **gc_kwargs)
    else:
        graph_conn_score = np.nan

    if scgraph_:
        print("scGraph...")
        scgraph_kwargs = convert_values(metric_kwargs.get("scgraph_kwargs", {}), scgraph)
        scgraph_avg, _, _, _ =  scgraph(adata_int, batch_key=batch_key, label_key=label_key, embed=embed, **scgraph_kwargs)
    else:
        scgraph_avg = np.nan

    if scgraphp_:
        print("scGraph with Fisher Poisson distance...")
        scgraphp_kwargs = convert_values(metric_kwargs.get("scgraphp_kwargs", {}), scgraph)
        scgraphp_avg, _, _, _ = scgraph(
            adata_int,
            batch_key=batch_key,
            label_key=label_key,
            embed=embed,
            method='fisher_poisson',
            **scgraphp_kwargs
        )
    else:
        scgraphp_avg = np.nan

    if scgraphm_:
        print("scGraph with Fisher multinomial distance...")
        scgraphm_kwargs = convert_values(metric_kwargs.get("scgraphm_kwargs", {}), scgraph)
        scgraphm_avg, _, _, _ = scgraph(
            adata_int,
            batch_key=batch_key,
            label_key=label_key,
            embed=embed,
            method='fisher_multinomial',
            **scgraphm_kwargs
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
        _, _, kbet_no_label_score = calc_kBET(
            adata_int, batch_key, embed, **kbet_kwargs
        )
    else:
        kbet_score = np.nan
        kbet_no_label_score = np.nan

    if not kbet_ and kbetall_:
        kbet_kwargs = convert_values(
            metric_kwargs.get("kbetall_kwargs", {}), calc_kBET
        )
        _, _, kbet_no_label_score = calc_kBET(
            adata_int, batch_key, embed, **kbet_kwargs
        )
    else:
        kbet_no_label_score = np.nan

    if clisi_:
        print("cLISI score...")
        clisi_kwargs = convert_values(metric_kwargs.get("clisi_kwargs", {}), clisi_graph)
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
        ilisi_kwargs = convert_values(metric_kwargs.get("ilisi_kwargs", {}), ilisi_graph)
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

    if hvg_:
        hvg_score = hvg_overlap(adata, adata_int, batch_key)
    else:
        hvg_score = np.nan

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
        ldfdiff_kwargs = convert_values(metric_kwargs.get('ldfdiff_kwargs', {}), ldf_diff)
        ldf_diff_score = ldf_diff(
            adata_pre=adata,
            adata_post=adata_int,
            batch_key=batch_key,
            embed_pre=embed_pre,
            embed_post=embed,
            **ldfdiff_kwargs
        )
    else:
        ldf_diff_score = np.nan

    if csmc_:
        print("Cell-specific mixing score...")
        csmc_kwargs = convert_values(metric_kwargs.get('csmc_kwargs', {}), cell_specific_mixing_score)
        cell_specific_mixing_score_result = cell_specific_mixing_score(
            adata_int, batch_key=batch_key, embed=embed, **csmc_kwargs
        )
    else:
        cell_specific_mixing_score_result = np.nan

    if align_:
        print("Alignment score...")
        align_kwargs = convert_values(metric_kwargs.get('align_kwargs', {}), alignment)
        alignment_score = alignment(adata_int, batch_key, embed, **align_kwargs)
    else:
        alignment_score = np.nan

    if agree_:
        print("Agreement score...")
        agree_kwargs = convert_values(metric_kwargs.get('agree_kwargs', {}), agreement)
        agreement_score = agreement(adata, adata_int, batch_key, label_key, embed, **agree_kwargs)
    else:
        agreement_score = np.nan

    if bknn_:
        print("Cross-batch k-NN score...")
        bknn_kwargs = convert_values(metric_kwargs.get('bknn_kwargs', {}), crossbatch_knn)
        crossbatch_knn_score = crossbatch_knn(adata_int, batch_key, label_key, embed, **bknn_kwargs)
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
        fknn_kwargs = convert_values(metric_kwargs.get("fknn_kwargs", {}), crossfold_knn)
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
        _, batch_ari_score = cluster_optimal_resolution(
            adata_int, label_key=batch_key, cluster_key=cluster_key, metric=batch_ari
        )
    else:
        batch_ari_score = np.nan

    if bps_ or all_cluster_:
        print("Batch purity score...")
        _, batch_purity_score = cluster_optimal_resolution(
            adata_int,
            label_key=batch_key,
            cluster_key=cluster_key,
            metric=batch_purity,
        )
    else:
        batch_purity_score = np.nan

    if bcentropy_ or all_cluster_:
        print("Batch Shannon entropy by cluster...")
        _, batch_shannon_entropy_score = cluster_optimal_resolution(
            adata_int,
            label_key=batch_key,
            cluster_key=cluster_key,
            metric=batch_shannon_entropy,
        )
    else:
        batch_shannon_entropy_score = np.nan

    if ccentropy_ or all_cluster_:
        print("Cell type Shannon entropy by cluster...")
        _, celltype_shannon_entropy_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=celltype_shannon_entropy
        )
    else:
        celltype_shannon_entropy_score = np.nan

    if bnentropy_ and cnentropy_:
        print("Shannon entropy nearest neighbor graph...")
        entropy_kwargs = convert_values(metric_kwargs.get("bnentropy_kwargs", {}), shannon_entropy_nn)
        batch_shannon_entropy_nn_score, cell_shannon_entropy_nn_score = (
            shannon_entropy_nn(
                adata_int,
                batch_key,
                label_key,
                embed=embed,
                **entropy_kwargs
            )
        )
    elif bnentropy_:
        print("Shannon entropy nearest neighbor graph...")
        entropy_kwargs = convert_values(metric_kwargs.get("bnentropy_kwargs", {}), shannon_entropy_nn)
        batch_shannon_entropy_nn_score, _ = shannon_entropy_nn(
            adata_int, batch_key, label_key, embed=embed, **entropy_kwargs
        )
        cell_shannon_entropy_nn_score = np.nan
    elif cnentropy_:
        print("Shannon entropy nearest neighbor graph...")
        entropy_kwargs = convert_values(
            metric_kwargs.get("cnentropy_kwargs", {}), shannon_entropy_nn
        )
        _, cell_shannon_entropy_nn_score = shannon_entropy_nn(
            adata_int, batch_key, label_key, embed=embed, **entropy_kwargs
        )
        batch_shannon_entropy_nn_score = np.nan
    else:
        batch_shannon_entropy_nn_score = cell_shannon_entropy_nn_score = np.nan

    if bnentropyw_ and cnentropyw_:
        print("Weighted Shannon entropy nearest neighbor graph...")
        entropy_kwargs = convert_values(
            metric_kwargs.get("bnentropyw_kwargs", {}), shannon_entropy_nn_weighted
        )
        bnentropyw_score, cnentropyw_score = (
            shannon_entropy_nn_weighted(
                adata_int, batch_key, label_key, embed=embed, **entropy_kwargs
            )
        )
    elif bnentropyw_:
        print("Weighted Shannon entropy nearest neighbor graph...")
        entropy_kwargs = convert_values(
            metric_kwargs.get("bnentropyw_kwargs", {}), shannon_entropy_nn_weighted
        )
        bnentropyw_score, _ = shannon_entropy_nn_weighted(
            adata_int, batch_key, label_key, embed=embed, **entropy_kwargs
        )
        cnentropyw_score = np.nan
    elif cnentropyw_:
        print("Weighted Shannon entropy nearest neighbor graph...")
        entropy_kwargs = convert_values(
            metric_kwargs.get("cnentropy_kwargs", {}), shannon_entropy_nn_weighted
        )
        _, cnentropyw_score = shannon_entropy_nn_weighted(
            adata_int, batch_key, label_key, embed=embed, **entropy_kwargs
        )
        bnentropyw_score = np.nan
    else:
        bnentropyw_score = cnentropyw_score = np.nan

    if arif_:
        print("ARI fixed...")
        arif_kwargs = metric_kwargs.get("arif_kwargs", {})
        if arif_kwargs.get('metric', 'euclidean') != 'euclidean':
            sc.pp.neighbors(adata_int, metric=arif_kwargs['metric'], use_rep=embed)
        arif_score = ari_fixed(adata_int, label_key, cluster_key, embed=embed)
    else:
        arif_score = np.nan

    if bari_ or all_cluster_:
        print("Balanced ARI...")
        _, bARI_score = cluster_optimal_resolution(
            adata_int, label_key=label_key, cluster_key=cluster_key, metric=bARI
        )
    else:
        bARI_score = np.nan

    if bami_ or all_cluster_:
        print("Balanced AMI...")
        _, bami_score = cluster_optimal_resolution(
            adata_int, label_key=label_key, cluster_key=cluster_key, metric=bAMI
        )
    else:
        bami_score = np.nan

    if bvm_ or all_cluster_:
        print("Balanced v-measure...")
        _, bv_measure_score = cluster_optimal_resolution(
            adata_int, label_key=label_key, cluster_key=cluster_key, metric=bv_measure
        )
    else:
        bv_measure_score = np.nan

    if probe_:
        print("Batch probing score...")
        bps_rf, bps_lr = batch_probing_score(adata_int, batch_key, embed)
    else:
        bps_rf = bps_lr = np.nan

    if cprobe_:
        print("Per cell type batch probing score...")
        cprobe_rf, cprobe_lr = per_label_batch_probing_score(adata_int, label_key, batch_key, embed)
    else:
        cprobe_rf = cprobe_lr = np.nan

    if ch_ or all_cluster_:
        print("Calinkski Harabasz score...")
        _, ch_score = cluster_optimal_resolution(
            adata_int,
            label_key=label_key,
            cluster_key=cluster_key,
            metric=calinski_harabasz,
            metric_kwargs={"embed": embed},
        )
    else:
        ch_score = np.nan
    
    if cch_:
        print("Cell type Calinski Harabasz...")
        cch_score = calinski_harabasz(adata_int, None, label_key, embed)
    else:
        cch_score = np.nan

    if clustacc_ or all_cluster_:
        print("Clustering accuracy...")
        _, clustacc_score = cluster_optimal_resolution(
            adata_int,
            label_key=label_key,
            cluster_key=cluster_key,
            metric=clustering_accuracy,
        )
    else:
        clustacc_score = np.nan

    if f1_ or all_cluster_:
        print("F1 score...")
        _, f1sm = cluster_optimal_resolution(
            adata_int,
            label_key=label_key,
            cluster_key=cluster_key,
            metric=f1_score_macro,
        )
    else:
        f1sm = np.nan

    if compact_ or all_cluster_:
        print("Compactness...")
        _, compactness_score = cluster_optimal_resolution(
            adata_int,
            label_key=label_key,
            cluster_key=cluster_key,
            metric=compactness,
            metric_kwargs={"embed": embed},
        )
    else:
        compactness_score = np.nan

    if corrmse_:
        print("Corr-MSE...")
        corrmse_score = corr_mse(adata, adata_int, batch_key, embed, embed_pre)
    else:
        corrmse_score = np.nan

    if kldiv_:
        print("KL divergence...")
        kldiv_kwargs = convert_values(metric_kwargs.get('kldiv_kwargs', {}), kl_divergence)
        kl_div_score = kl_divergence(adata_int, batch_key, embed, **kldiv_kwargs)
    else:
        kl_div_score = np.nan

    if homo_ or all_cluster_:
        print("Homogeneity...")
        _, homogeneity_score = cluster_optimal_resolution(
            adata_int,
            label_key,
            cluster_key,
            metric=homogeneity,
        )
    else:
        homogeneity_score = np.nan

    if fmi_ or all_cluster_:
        print("Fowlkes Mallows...")
        _, fm_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=fowlkes_mallows
        )
    else:
        fm_score = np.nan

    if jacc_ or all_cluster_:
        print("Jaccard index...")
        _, jaccard_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=jaccard_index
        )
    else:
        jaccard_score = np.nan

    if mcc_ or all_cluster_:
        print("Matthews correlation coefficient")
        _, mcc_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=matthews_corrcoef
        )
    else:
        mcc_score = np.nan

    if modularity_ or all_cluster_:
        print("Modularity...")
        _, modularity_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=modularity
        )
    else:
        modularity_score = np.nan

    if adgbd_:
        print("Average deviation from global batch distribution...")
        adgbd_kwargs = convert_values(metric_kwargs.get('adgbd_kwargs', {}), adgbd)
        adgbd_score = adgbd(adata_int, batch_key, embed=embed, **adgbd_kwargs)
    else:
        adgbd_score = np.nan

    if awc_ or all_cluster_:
        print("Adjusted Wallace score...")
        _, aw_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=adjusted_wallace_score
        )
    else:
        aw_score = np.nan

    if ami_ or all_cluster_:
        print("Adjusted mutual information")
        _, ami_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=ami
        )
    else:
        ami_score = np.nan

    if bras_:
        print("BRAS...")
        bras_kwargs = convert_values(metric_kwargs.get('bras_kwargs', {}), bras)
        bras_score = bras(adata_int, label_key, batch_key, embed, **bras_kwargs)
    else:
        bras_score = np.nan

    if cbe_ or all_cluster_:
        print("Cluster batch entropy...")
        _, cluster_batch_entropy_score = cluster_optimal_resolution(
            adata_int, batch_key, cluster_key, metric=cluster_batch_entropy
        )
    else:
        cluster_batch_entropy_score = np.nan

    if comp_ or all_cluster_:
        print("Completeness...")
        _, completeness_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=completeness
        )
    else:
        completeness_score = np.nan

    if dbi_:
        print("Cell type Davies-Bouldin...")
        cdbi_score = davies_bouldin(adata_int, None, label_key, embed)
    else:
        cdbi_score = np.nan

    if clustdbi_ or all_cluster_:
        print("Cluster Davies-Bouldin...")
        _, clust_dbi_score = cluster_optimal_resolution(
            adata_int,
            label_key,
            cluster_key,
            metric=davies_bouldin,
            metric_kwargs={"embed": embed},
        )
    else:
        clust_dbi_score = np.nan

    if knndiv_:
        print("KNN divergence...")
        knn_div_score = knn_divergence(adata_int, label_key, batch_key, embed)
    else:
        knn_div_score = np.nan

    if ksim_:
        print("kSIM...")
        ksim_kwargs = convert_values(metric_kwargs.get('ksim_kwargs', {}), ksim)
        ksim_score = ksim(adata_int, label_key, embed, **ksim_kwargs)
    else:
        ksim_score = np.nan

    if mix_:
        print("Mixing score...")
        mix_kwargs = convert_values(metric_kwargs.get('mix_kwargs', {}), mixing)
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
        if nmif_kwargs.get('metric', 'euclidean') != 'euclidean':
            sc.pp.neighbors(adata_int, metric=nmif_kwargs['metric'], use_rep=embed)
        nmi_fixed_score = nmi_fixed(adata_int, label_key, cluster_key)
    else:
        nmi_fixed_score = np.nan

    if nnmap_:
        print("Nearest neighbor mean average precision")
        nnmap_kwargs = convert_values(metric_kwargs.get('nnmap_kwargs', {}), nnmap)
        nnmap_score = nnmap(adata_int, label_key, embed, **nnmap_kwargs)
    else:
        nnmap_score = np.nan

    if ocs_:
        print("Over correction...")
        ocs_kwargs = convert_values(metric_kwargs.get('ocs_kwargs', {}), over_correction)
        oc_score = over_correction(adata_int, label_key, embed=embed, **ocs_kwargs)
    else:
        oc_score = np.nan

    if pptpc_:
        print("Positive of positive and true positive cells...")
        pptpc_kwargs = convert_values(metric_kwargs.get('pptpc_kwargs', {}), pptpc)
        pptpc_score = pptpc(adata_int, batch_key, label_key, embed=embed, **pptpc_kwargs)
    else:
        pptpc_score = np.nan

    if pstdb_:
        print("Proportion of same cell type but different batch...")
        pstdb_kwargs = convert_values(metric_kwargs.get('pstdb_kwargs', {}), pstdb)
        pstdb_score = pstdb(adata_int, batch_key, label_key, embed, **pstdb_kwargs)
    else:
        pstdb_score = np.nan

    if purity_ or all_cluster_:
        print("Purity...")
        _, purity_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=purity
        )
    else:
        purity_score = np.nan

    if rknn_:
        print("Retained KNN...")
        rknn_kwargs = convert_values(metric_kwargs.get('rknn_kwargs', {}), retained_knn)
        retained_knn_score = retained_knn(adata, adata_int, batch_key, embed, **rknn_kwargs)
    else:
        retained_knn_score = np.nan

    if sv_:
        print("Scaled variance...")
        sv_score = scaled_variance(adata_int, batch_key, embed)
    else:
        sv_score = np.nan

    if separation_ or all_cluster_:
        _, separation_score = cluster_optimal_resolution(
            adata_int,
            label_key,
            cluster_key,
            metric=separation,
            metric_kwargs={"embed": embed},
        )
    else:
        separation_score = np.nan

    if vm_ or all_cluster_:
        print("V measure...")
        _, v_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=v_measure
        )
    else:
        v_score = np.nan

    if vi_ or all_cluster_:
        print("Variation of information...")
        _, vi_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=variation_of_information
        )
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
        "PROBEMIN": np.min([bps_rf, bps_lr]),
        "CPROBE": (cprobe_rf + cprobe_lr) / 2,
        "CPROBERF": cprobe_rf,
        "CPROBELR": cprobe_lr,
        "BATCHARI": batch_ari_score,
        "BPS": batch_purity_score,
        "BCENTROPY": batch_shannon_entropy_score,
        "BRAS": bras_score,
        "CH": ch_score,
        "CCC": cc_score,
        "CCH": cch_score,
        "CSMC": cell_specific_mixing_score_result,
        "CCENTROPY": celltype_shannon_entropy_score,
        "CLISI": clisi_score,
        "CILISI": cilisi_score,
        "CLUSTASW": silhouette_cluster_score,
        "CLUSTACC": clustacc_score,
        "CLUSTDBI": clust_dbi_score,
        "CBE": cluster_batch_entropy_score,
        "COMPACT": compactness_score,
        "COMP": completeness_score,
        "CORRMSE": corrmse_score,
        "BKNN": crossbatch_knn_score,
        "BLOGREG": crossbatch_logistic_score,
        "FKNN": crossfold_knn_score,
        "FLOGREG": crossfold_logistic_score,
        "DBI": cdbi_score,
        "F1": f1sm,
        "FMI": fm_score,
        "GC": graph_conn_score,
        "HOMO": homogeneity_score,
        "HVG": hvg_score,
        "ILISI": ilisi_score,
        "ISOF1": il_score_f1,
        "ISOASW": il_score_asw,
        "ISOLISI": il_score_lisi,
        "JACC": jaccard_score,
        "KBET": kbet_score,
        "KBETALL": kbet_no_label_score,
        "KLDIV": kl_div_score,
        "KNNDIV": knn_div_score,
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
        "BNENTROPY": batch_shannon_entropy_nn_score,
        "CNENTROPY": cell_shannon_entropy_nn_score,
        "BNENTROPYW": bnentropyw_score,
        "CNENTROPYW": cnentropyw_score,
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
                        converted_kwargs[key] = value.lower() == 'true'
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

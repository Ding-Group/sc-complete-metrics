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
from .batch_probing_score import batch_probing_score
from .batch_purity_score import batch_purity_score
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
from .isolated_labels import isolated_labels
from .jaccard import jaccard_index
from .kbet import kBET
from .kl_divergence import kl_divergence
from .knn_divergence import knn_divergence
from .ldf_diff import ldf_diff
from .lisi import clisi_graph, ilisi_graph
from .matthews_corrcoef import matthews_corrcoef
from .mixing import mixing
from .modularity import modularity
from .morans_i import morans_i
from .nmi import nmi, nmi_fixed
from .over_correction import over_correction
from .pcr import pcr_comparison
from .pptpc import pptpc
from .purity import purity
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
    bps_=False,
    bras_=False,
    bvm_=False,
    ch_=False,
    ccc_=False,
    csmc_=False,
    cbe_=False,
    cilisi_=False,
    clustacc_=False,
    compact_=False,
    comp_=False,
    bknn_=False,
    blogreg_=False,
    fknn_=False,
    flogreg_=False,
    dbi_=False,
    f1_=False,
    fmi_=False,
    gc_=False,
    hvg_=False,
    homo_=False,
    isof1_=False,
    isoasw_=False,
    jacc_=False,
    kbet_=False,
    kldiv_=False,
    knndiv_=False,
    ldfdiff_=False,
    clisi_=False,
    ilisi_=False,
    mcc_=False,
    mix_=False,
    modularity_=False,
    morani_=False,
    nmi_=False,
    nmif_=False,
    ocs_=False,
    pcr_=False,
    pptpc_=False,
    purity_=False,
    rknn_=False,
    scgraph_=False,
    sv_=False,
    separation_=False,
    bcentropy_=False,
    ccentropy_=False,
    bnentropy_=False,
    cnentropy_=False,
    casw_=False,
    basw_=False,
    clustasw_=False,
    traj_=False,
    vm_=False,
    vi_=False,
    metric_kwargs = dict(),
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

    if nmi_:
        print("NMI...")
        _, nmi_score = cluster_optimal_resolution(
            adata_int, label_key=label_key, cluster_key=cluster_key, metric=nmi
        )
    else:
        nmi_score = np.nan

    if ari_:
        print("ARI...")
        _, ari_score = cluster_optimal_resolution(
            adata_int, label_key=label_key, cluster_key=cluster_key, metric=ari
        )
    else:
        ari_score = np.nan

    if casw_:
        print("Cell type silhouette score...")
        asw_label = silhouette(
            adata_int,
            label_key=label_key,
            embed=embed,
            metric=metric_kwargs.get("si_metric", "euclidean"),
        )
    else:
        asw_label = np.nan

    if basw_:
        print("Batch silhouette score...")
        asw_batch = silhouette_batch(
            adata_int,
            batch_key=batch_key,
            label_key=label_key,
            embed=embed,
            metric=metric_kwargs.get("si_metric", "euclidean"),
            return_all=False,
            verbose=False,
        )
    else:
        asw_batch = np.nan

    if clustasw_:
        print("Cluster ASW score...")
        _, silhouette_cluster_score = cluster_optimal_resolution(
            adata_int,
            label_key,
            cluster_key,
            metric=silhouette_cluster,
            metric_kwargs={"embed": embed},
        )
    else:
        silhouette_cluster_score = np.nan

    if pcr_:
        print("PC regression...")
        pcr_score = pcr_comparison(
            adata, adata_int, embed=embed, covariate=batch_key, verbose=verbose
        )
    else:
        pcr_score = np.nan

    if ccc_:
        print("cell cycle effect...")
        cc_score = cell_cycle(
            adata,
            adata_int,
            batch_key=batch_key,
            embed=embed,
            agg_func=np.mean,
            organism=organism,
        )
    else:
        cc_score = np.nan

    if isof1_:
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
        il_score_asw = isolated_labels(
            adata_int,
            label_key=label_key,
            batch_key=batch_key,
            embed=embed,
            cluster=False,
            iso_threshold=metric_kwargs.get("n_isolated", None),
            verbose=False,
        )
    else:
        il_score_asw = np.nan

    if gc_:
        print("Graph connectivity...")
        graph_conn_score = graph_connectivity(adata_int, label_key=label_key)
    else:
        graph_conn_score = np.nan

    if scgraph_:
        print("scGraph connectivity classifier...")
        scgraph_avg, scgraph_rank_pca, scgraph_corr_pca, scgraph_corr_weighted = (
            scgraph(adata_int, batch_key=batch_key, label_key=label_key, embed=embed)
        )
    else:
        scgraph_avg = scgraph_rank_pca = scgraph_corr_pca = scgraph_corr_weighted = (
            np.nan
        )

    if kbet_:
        print("kBET...")
        kbet_score = kBET(
            adata_int,
            batch_key=batch_key,
            label_key=label_key,
            type_=metric_kwargs.get("type_", "knn"),
            embed=embed,
            scaled=True,
            verbose=verbose,
        )
    else:
        kbet_score = np.nan

    if clisi_:
        print("cLISI score...")
        clisi_score = clisi_graph(
            adata_int,
            batch_key=batch_key,
            label_key=label_key,
            type_=metric_kwargs.get("type_", "knn"),
            subsample=metric_kwargs.get("subsample", 0.5) * 100,
            scale=True,
            n_cores=metric_kwargs.get("n_cores", 1),
            verbose=verbose,
        )
    else:
        clisi_score = np.nan

    if ilisi_:
        print("iLISI score...")
        ilisi_score = ilisi_graph(
            adata_int,
            batch_key=batch_key,
            type_=metric_kwargs.get("type_", "knn"),
            subsample=metric_kwargs.get("subsample", 0.5) * 100,
            scale=True,
            n_cores=metric_kwargs.get("n_cores", 1),
            verbose=verbose,
        )
    else:
        ilisi_score = np.nan

    if cilisi_:
        print("CiLISI score...")
        cilisi_score = cilisi(adata_int, batch_key, label_key, embed)
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
        ldf_diff_score = ldf_diff(
            adata_pre=adata,
            adata_post=adata_int,
            batch_key=batch_key,
            embed_pre=embed_pre,
            embed_post=embed,
        )
    else:
        ldf_diff_score = np.nan

    if csmc_:
        print("Cell-specific mixing score...")
        cell_specific_mixing_score_result = cell_specific_mixing_score(
            adata_int, batch_key=batch_key, embed=embed
        )
    else:
        cell_specific_mixing_score_result = np.nan

    if align_:
        print("Alignment score…")
        alignment_score = alignment(adata_int, batch_key, embed)
    else:
        alignment_score = np.nan

    if agree_:
        print("Agreement score...")
        agreement_score = agreement(adata, adata_int, batch_key, label_key, embed)
    else:
        agreement_score = np.nan

    if bknn_:
        print("Cross-batch k-NN score...")
        crossbatch_knn_score = crossbatch_knn(adata_int, batch_key, label_key, embed)
    else:
        crossbatch_knn_score = np.nan

    if blogreg_:
        print("Cross-batch logistic regression...")
        crossbatch_logistic_score = crossbatch_logistic(
            adata_int,
            label_key,
            batch_key,
            embed,
        )
    else:
        crossbatch_logistic_score = np.nan

    if fknn_:
        print("Cross-fold k-NN score...")
        crossfold_knn_score = crossfold_knn(adata_int, label_key, embed)
    else:
        crossfold_knn_score = np.nan

    if flogreg_:
        print("Cross-fold logistic regression...")
        crossfold_logistic_score = crossfold_logistic(adata_int, label_key, embed)
    else:
        crossfold_logistic_score = np.nan

    if batchari_:
        print("Batch ARI...")
        _, batch_ari_score = cluster_optimal_resolution(
            adata_int, label_key=batch_key, cluster_key=cluster_key, metric=batch_ari
        )
    else:
        batch_ari_score = np.nan

    if bps_:
        print("Batch purity score...")
        _, batch_purity = cluster_optimal_resolution(
            adata_int,
            label_key=batch_key,
            cluster_key=cluster_key,
            metric=batch_purity_score,
        )
    else:
        batch_purity = np.nan

    if bcentropy_:
        print("Batch Shannon entropy...")
        _, batch_shannon_entropy_score = cluster_optimal_resolution(
            adata_int,
            label_key=batch_key,
            cluster_key=cluster_key,
            metric=batch_shannon_entropy,
        )
    else:
        batch_shannon_entropy_score = np.nan

    if ccentropy_:
        print("Cell type Shannon entropy...")
        _, celltype_shannon_entropy_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=celltype_shannon_entropy
        )
    else:
        celltype_shannon_entropy_score = np.nan

    if bnentropy_ and cnentropy_:
        print("Shannon entropy nearest neighbor graph...")
        batch_shannon_entropy_nn_score, cell_shannon_entropy_nn_score = (
            shannon_entropy_nn(adata_int, batch_key, label_key, embed=embed)
        )
    elif bnentropy_:
        print("Shannon entropy nearest neighbor graph...")
        batch_shannon_entropy_nn_score, _ = shannon_entropy_nn(
            adata_int, batch_key, label_key, embed=embed
        )
    elif cnentropy_:
        print("Shannon entropy nearest neighbor graph...")
        _, cell_shannon_entropy_nn_score = shannon_entropy_nn(
            adata_int, batch_key, label_key, embed=embed
        )
    else:
        batch_shannon_entropy_nn_score = cell_shannon_entropy_nn_score = np.nan

    if arif_:
        print("ARI fixed...")
        arif_score = ari_fixed(adata_int, label_key, cluster_key, embed=embed)
    else:
        arif_score = np.nan

    if bari_:
        print("Balanced ARI...")
        _, bARI_score = cluster_optimal_resolution(
            adata_int, label_key=label_key, cluster_key=cluster_key, metric=bARI
        )
    else:
        bARI_score = np.nan

    if bami_:
        print("Balanced AMI...")
        _, bami_score = cluster_optimal_resolution(
            adata_int, label_key=label_key, cluster_key=cluster_key, metric=bAMI
        )
    else:
        bami_score = np.nan

    if bvm_:
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

    if ch_:
        print("Calinkski Harabasz score")
        _, ch_score = cluster_optimal_resolution(
            adata_int,
            label_key=label_key,
            cluster_key=cluster_key,
            metric=calinski_harabasz,
            metric_kwargs={"embed": embed},
        )
    else:
        ch_score = np.nan

    if clustacc_:
        print("Clustering accuracy...")
        _, clustacc_score = cluster_optimal_resolution(
            adata_int,
            label_key=label_key,
            cluster_key=cluster_key,
            metric=clustering_accuracy,
        )
    else:
        clustacc_score = np.nan

    if f1_:
        print("F1 score...")
        _, f1sm = cluster_optimal_resolution(
            adata_int,
            label_key=label_key,
            cluster_key=cluster_key,
            metric=f1_score_macro,
        )
    else:
        f1sm = np.nan

    if compact_:
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

    if kldiv_:
        print("KL divergence...")
        kl_div_score = kl_divergence(adata_int, batch_key, embed)
    else:
        kl_div_score = np.nan

    if homo_:
        print("Homogeneity...")
        _, homogeneity_score = cluster_optimal_resolution(
            adata_int,
            label_key,
            cluster_key,
            metric=homogeneity,
        )
    else:
        homogeneity_score = np.nan

    if fmi_:
        print("Fowlkes Mallows...")
        _, fm_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=fowlkes_mallows
        )
    else:
        fm_score = np.nan

    if jacc_:
        print("Jaccard index...")
        _, jaccard_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=jaccard_index
        )
    else:
        jaccard_score = np.nan

    if mcc_:
        print("Matthews correlation coefficient")
        _, mcc_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=matthews_corrcoef
        )
    else:
        mcc_score = np.nan

    if modularity_:
        print("Modularity...")
        _, modularity_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=modularity
        )
    else:
        modularity_score = np.nan

    if adgbd_:
        print("Average deviation from global batch distribution...")
        adgbd_score = adgbd(adata_int, batch_key, embed=embed)
    else:
        adgbd_score = np.nan

    if awc_:
        print("Adjusted Wallace score...")
        _, aw_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=adjusted_wallace_score
        )
    else:
        aw_score = np.nan

    if ami_:
        print("Adjusted mutual information")
        _, ami_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=ami
        )
    else:
        ami_score = np.nan

    if bras_:
        print("BRAS...")
        bras_score = bras(adata_int, label_key, batch_key, embed)
    else:
        bras_score = np.nan

    if cbe_:
        print("Cluster batch entropy...")
        _, cluster_batch_entropy_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=cluster_batch_entropy
        )
    else:
        cluster_batch_entropy_score = np.nan

    if comp_:
        print("Completeness...")
        _, completeness_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=completeness
        )
    else:
        completeness_score = np.nan

    if dbi_:
        print("Davies-Bouldin...")
        davies_bouldin_score = davies_bouldin(adata_int, label_key, embed)
    else:
        davies_bouldin_score = np.nan

    if knndiv_:
        print("KNN divergence...")
        knn_div_score = knn_divergence(adata_int, label_key, batch_key, embed)
    else:
        knn_div_score = np.nan

    if mix_:
        print("Mixing score...")
        mixing_score = mixing(adata_int, batch_key, embed)
    else:
        mixing_score = np.nan

    if morani_:
        print("Moran's I...")
        moransi_score = morans_i(adata, adata_int, label_key, batch_key, "X_pca", embed)
    else:
        moransi_score = np.nan

    if nmif_:
        print("NMI fixed...")
        nmi_fixed_score = nmi_fixed(adata_int, label_key, cluster_key)
    else:
        nmi_fixed_score = np.nan

    if ocs_:
        print("Over correction...")
        oc_score = over_correction(adata_int, label_key, embed=embed)
    else:
        oc_score = np.nan

    if pptpc_:
        print("Positive of positive and true positive cells...")
        pptpc_score = pptpc(adata_int, batch_key, label_key, embed=embed)
    else:
        pptpc_score = np.nan

    if purity_:
        print("Purity...")
        _, purity_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=purity
        )
    else:
        purity_score = np.nan

    if rknn_:
        print("Retained KNN...")
        retained_knn_score = retained_knn(adata, adata_int, batch_key, embed)
    else:
        retained_knn_score = np.nan

    if sv_:
        print("Scaled variance...")
        sv_score = scaled_variance(adata_int, batch_key, embed)
    else:
        sv_score = np.nan

    if separation_:
        _, separation_score = cluster_optimal_resolution(
            adata_int,
            label_key,
            cluster_key,
            metric=separation,
            metric_kwargs={"embed": embed},
        )
    else:
        separation_score = np.nan

    if vm_:
        print("V measure...")
        _, v_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=v_measure
        )
    else:
        v_score = np.nan

    if vi_:
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
        "BATCHARI": batch_ari_score,
        "BPS": batch_purity,
        "BCENTROPY": batch_shannon_entropy_score,
        "BRAS": bras_score,
        "CH": ch_score,
        "CCC": cc_score,
        "CSMC": cell_specific_mixing_score_result,
        "CCENTROPY": celltype_shannon_entropy_score,
        "CLISI": clisi_score,
        "CILISI": cilisi_score,
        "CLUSTASW": silhouette_cluster_score,
        "CLUSTACC": clustacc_score,
        "CBE": cluster_batch_entropy_score,
        "COMPACT": compactness_score,
        "COMPLETE": completeness_score,
        "BKNN": crossbatch_knn_score,
        "BLOGREG": crossbatch_logistic_score,
        "FKNN": crossfold_knn_score,
        "FLOGREG": crossfold_logistic_score,
        "DBI": davies_bouldin_score,
        "F1": f1sm,
        "FMI": fm_score,
        "GC": graph_conn_score,
        "HOMO": homogeneity_score,
        "HVG": hvg_score,
        "ILISI": ilisi_score,
        "ISOF1": il_score_f1,
        "ISOASW": il_score_asw,
        "JACC": jaccard_score,
        "KBET": kbet_score,
        "KLDIV": kl_div_score,
        "KNNDIV": knn_div_score,
        "LDFDIFF": ldf_diff_score,
        "NMI": nmi_score,
        "MCC": mcc_score,
        "MIX": mixing_score,
        "MODULARITY": modularity_score,
        "MORANI": moransi_score,
        "NMIF": nmi_fixed_score,
        "OCS": oc_score,
        "PCR": pcr_score,
        "PPTPC": pptpc_score,
        "PURITY": purity_score,
        "RKNN": retained_knn_score,
        "SV": sv_score,
        "SCGRAPH": scgraph_avg,
        "SEPARATION": separation_score,
        "BNENTROPY": batch_shannon_entropy_nn_score,
        "CNENTROPY": cell_shannon_entropy_nn_score,
        "CASW": asw_label,
        "TRAJ": trajectory_score,
        "VM": v_score,
        "VI": vi_score,
        # "scGraph_corr_pca": scgraph_corr_pca,
        # "scGraph_corr_weighted": scgraph_corr_weighted,
        # "scGraph_rank_pca": scgraph_rank_pca,
    }

    return pd.DataFrame.from_dict(results, orient="index")

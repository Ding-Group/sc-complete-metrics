import numpy as np
import pandas as pd
import scanpy as sc

from ..utils import check_adata, check_batch
from .agreement import agreement
from .alignment import alignment
from .ari import ari
from .bAMI import bAMI
from .bARI import bARI
from .batch_ari import batch_ari
from .batch_probing_score import batch_probing_score
from .batch_purity_score import batch_purity_score
from .bv_measure import bv_measure
from .calinski_harabasz import calinski_harabasz
from .cell_cycle import cell_cycle
from .cell_specific_mixing_score import cell_specific_mixing_score
from .clustering import cluster_optimal_resolution
from .clustering_accuracy import clustering_accuracy
from .compactness import compactness
from .crossbatch_knn import crossbatch_knn
from .crossbatch_logistic import crossbatch_logistic
from .crossfold_knn import crossfold_knn
from .crossfold_logistic import crossfold_logistic
from .f1_score import f1_score_macro
from .fowlkes_mallows import fowlkes_mallows
from .graph_connectivity import graph_connectivity
from .highly_variable_genes import hvg_overlap
from .homogeneity import homogeneity
from .isolated_labels import isolated_labels
from .jaccard import jaccard_index
from .kbet import kBET
from .kl_divergence import kl_divergence
# from .ldf_diff import ldf_diff
from .lisi import clisi_graph, ilisi_graph
from .matthews_corrcoef import matthews_corrcoef
from .modularity import modularity
from .nmi import nmi
from .pcr import pcr_comparison
from .sc_graph import scgraph
from .shannon_entropy_cluster import batch_shannon_entropy, celltype_shannon_entropy
from .silhouette import silhouette, silhouette_batch
from .trajectory import trajectory_conservation


def metrics_fast(adata, adata_int, batch_key, label_key, **kwargs):
    return metrics(
        adata,
        adata_int,
        batch_key,
        label_key,
        isolated_labels_asw_=True,
        silhouette_=True,
        hvg_score_=True,
        graph_conn_=True,
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
        hvg_score_=True,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=True,
        trajectory_=True,
        nmi_=True,
        ari_=True,
        cell_cycle_=True,
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
        hvg_score_=True,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=True,
        trajectory_=True,
        nmi_=True,
        ari_=True,
        cell_cycle_=True,
        kBET_=True,
        ilisi_=True,
        clisi_=True,
        sc_graph_=True,
        ldf_diff_=True,
        cell_specific_mixing_score_=True,
        crossbatch_knn_=True,
        crossbatch_logistic_=True,
        batch_ari_=True,
        batch_purity_=True,
        batch_shannon_entropy_=True,
        **kwargs,
    )


def metrics(
    adata,
    adata_int,
    batch_key,
    label_key,
    embed="X_pca",
    cluster_key="cluster",
    ari_=False,
    nmi_=False,
    silhouette_=False,
    si_metric="euclidean",
    pcr_=False,
    cell_cycle_=False,
    organism="mouse",
    hvg_score_=False,
    isolated_labels_=False,
    isolated_labels_f1_=False,
    isolated_labels_asw_=False,
    n_isolated=None,
    graph_conn_=False,
    trajectory_=False,
    kBET_=False,
    lisi_graph_=False,
    ilisi_=False,
    clisi_=False,
    subsample=0.5,
    n_cores=1,
    type_=None,
    verbose=False,
    alignment_=False,
    agreement_=False,
    ldf_diff_=False,
    cell_specific_mixing_score_=False,
    crossbatch_knn_=False,
    sc_graph_=False,
    crossbatch_logistic_=False,
    batch_ari_=False,
    batch_purity_=False,
    batch_shannon_entropy_=False,
    bARI_=False,
    bAMI_=False,
    bv_measure_=False,
    batch_probing_score_=False,
    calinski_harabasz_=False,
    clustering_accuracy_=False,
    crossfold_logistic_=False,
    crossfold_knn_=False,
    f1_score_macro_=False,
    compactness_=False,
    kl_divergence_=False,
    homogeneity_=False,
    fowlkes_mallows_=False,
    jaccard_=False,
    matthews_corrcoef_=False,
    modularity_=False,
):
    check_adata(adata)
    check_batch(batch_key, adata.obs)
    check_batch(label_key, adata.obs)

    check_adata(adata_int)
    check_batch(batch_key, adata_int.obs)
    check_batch(label_key, adata_int.obs)

    # Run `sc.pp.neighbors` to get the kNN graph
    sc.pp.neighbors(adata_int, use_rep=embed)

    # TODO: make the clustering based metrics independent of each other.
    # Each should find the clustering with the highest metric value.
    # clustering

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

    if silhouette_:
        print("Silhouette score...")
        asw_label = silhouette(
            adata_int, label_key=label_key, embed=embed, metric=si_metric
        )
        asw_batch = silhouette_batch(
            adata_int,
            batch_key=batch_key,
            label_key=label_key,
            embed=embed,
            metric=si_metric,
            return_all=False,
            verbose=False,
        )
    else:
        asw_label = np.nan
        asw_batch = np.nan

    if pcr_:
        print("PC regression...")
        pcr_score = pcr_comparison(
            adata, adata_int, embed=embed, covariate=batch_key, verbose=verbose
        )
    else:
        pcr_score = np.nan

    if cell_cycle_:
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

    if isolated_labels_f1_ or isolated_labels_:
        print("Isolated labels F1...")
        il_score_f1 = isolated_labels(
            adata_int,
            label_key=label_key,
            batch_key=batch_key,
            embed=embed,
            cluster=True,
            iso_threshold=n_isolated,
            verbose=False,
        )
    else:
        il_score_f1 = np.nan

    if isolated_labels_asw_ or isolated_labels_:
        print("Isolated labels ASW...")
        il_score_asw = (
            isolated_labels(
                adata_int,
                label_key=label_key,
                batch_key=batch_key,
                embed=embed,
                cluster=False,
                iso_threshold=n_isolated,
                verbose=False,
            )
            if silhouette_
            else np.nan
        )
    else:
        il_score_asw = np.nan

    if graph_conn_:
        print("Graph connectivity...")
        graph_conn_score = graph_connectivity(adata_int, label_key=label_key)
    else:
        graph_conn_score = np.nan

    if sc_graph_:
        print("scGraph connectivity classifier...")
        scgraph_avg, scgraph_rank_pca, scgraph_corr_pca, scgraph_corr_weighted = (
            scgraph(adata_int, batch_key=batch_key, label_key=label_key, embed=embed)
        )
    else:
        scgraph_avg = scgraph_rank_pca = scgraph_corr_pca = scgraph_corr_weighted = (
            np.nan
        )

    if kBET_:
        print("kBET...")
        kbet_score = kBET(
            adata_int,
            batch_key=batch_key,
            label_key=label_key,
            type_=type_,
            embed=embed,
            scaled=True,
            verbose=verbose,
        )
    else:
        kbet_score = np.nan

    if lisi_graph_:
        clisi_ = True
        ilisi_ = True

    if clisi_:
        print("cLISI score...")
        clisi = clisi_graph(
            adata_int,
            batch_key=batch_key,
            label_key=label_key,
            type_=type_,
            subsample=subsample * 100,
            scale=True,
            n_cores=n_cores,
            verbose=verbose,
        )
    else:
        clisi = np.nan

    if ilisi_:
        print("iLISI score...")
        ilisi = ilisi_graph(
            adata_int,
            batch_key=batch_key,
            type_=type_,
            subsample=subsample * 100,
            scale=True,
            n_cores=n_cores,
            verbose=verbose,
        )
    else:
        ilisi = np.nan

    if hvg_score_:
        hvg_score = hvg_overlap(adata, adata_int, batch_key)
    else:
        hvg_score = np.nan

    if trajectory_:
        print("Trajectory conservation score...")
        trajectory_score = trajectory_conservation(
            adata,
            adata_int,
            label_key=label_key,
        )
    else:
        trajectory_score = np.nan

    if ldf_diff_:
        print("LDF difference score...")
        ldf_diff_score = ldf_diff(
            adata,
            batch_key=batch_key,
            embed=embed,
            verbose=verbose,
        )
    else:
        ldf_diff_score = np.nan

    if cell_specific_mixing_score_:
        print("Cell-specific mixing score...")
        cell_specific_mixing_score_result = cell_specific_mixing_score(
            adata,
            batch_key=batch_key,
            embed=embed,
            verbose=verbose,
        )
    else:
        cell_specific_mixing_score_result = np.nan

    if alignment_:
        print("Alignment score…")
        alignment_score = alignment(adata_int, batch_key, embed)
    else:
        alignment_score = np.nan

    if agreement_:
        print("Agreement score...")
        agreement_score = agreement(adata, adata_int, batch_key, label_key, embed)
    else:
        agreement_score = np.nan

    if crossbatch_knn_:
        print("Cross-batch k-NN score...")
        crossbatch_knn_score = crossbatch_knn(adata, batch_key, label_key, embed)
    else:
        crossbatch_knn_score = np.nan

    if crossbatch_logistic_:
        print("Cross-batch logistic regression...")
        crossbatch_logistic_score = crossbatch_logistic(
            adata_int,
            label_key,
            batch_key,
            embed,
        )
    else:
        crossbatch_logistic_score = np.nan

    if crossfold_knn_:
        print("Cross-fold k-NN score...")
        crossfold_knn_score = crossfold_knn(adata_int, label_key, embed)
    else:
        crossfold_knn_score = np.nan

    if crossfold_logistic_:
        print("Cross-fold logistic regression...")
        crossfold_logistic_score = crossfold_logistic(adata_int, label_key, embed)
    else:
        crossfold_logistic_score = np.nan

    if batch_ari_:
        print("Batch ARI...")
        batch_ari_score = cluster_optimal_resolution(
            adata_int, label_key=batch_key, cluster_key=cluster_key, metric=batch_ari
        )
    else:
        batch_ari_score = np.nan

    if batch_purity_:
        print("Batch purity score...")
        batch_purity = cluster_optimal_resolution(
            adata_int,
            label_key=batch_key,
            cluster_key=cluster_key,
            metric=batch_purity_score,
        )
    else:
        batch_purity = np.nan

    if batch_shannon_entropy_:
        print("Batch Shannon entropy...")
        batch_shannon_entropy_score = cluster_optimal_resolution(
            adata_int,
            label_key=batch_key,
            cluster_key=cluster_key,
            metric=batch_shannon_entropy,
        )
    else:
        batch_shannon_entropy_score = np.nan

    if bARI_:
        print("Balanced ARI")
        _, bARI_score = cluster_optimal_resolution(
            adata_int, label_key=label_key, cluster_key=cluster_key, metric=bARI
        )
    else:
        bARI_score = np.nan

    if bAMI_:
        print("Balanced AMI")
        _, bAMI_score = cluster_optimal_resolution(
            adata_int, label_key=label_key, cluster_key=cluster_key, metric=bAMI
        )
    else:
        bAMI_score = np.nan

    if bv_measure_:
        print("Balanced v-measure")
        _, bv_measure_score = cluster_optimal_resolution(
            adata_int, label_key=label_key, cluster_key=cluster_key, metric=bv_measure
        )
    else:
        bv_measure_score = np.nan

    if batch_probing_score_:
        print("Batch probing score...")
        bps_rf, bps_lr = batch_probing_score(adata_int, batch_key, embed)
    else:
        bps_rf = bps_lr = np.nan

    if calinski_harabasz_:
        print("Calinkski Harabasz score")
        ch_score = cluster_optimal_resolution(
            adata_int,
            label_key=label_key,
            cluster_key=cluster_key,
            metric=calinski_harabasz,
        )
    else:
        ch_score = np.nan

    if clustering_accuracy_:
        print("Clustering accuracy...")
        clustacc_score = cluster_optimal_resolution(
            adata_int,
            label_key=label_key,
            cluster_key=cluster_key,
            metric=clustering_accuracy,
        )
    else:
        clustacc_score = np.nan

    if f1_score_macro_:
        print("F1 score...")
        f1sm = cluster_optimal_resolution(
            adata_int,
            label_key=label_key,
            cluster_key=cluster_key,
            metric=f1_score_macro,
        )
    else:
        f1sm = np.nan

    if compactness_:
        print("Compactness...")
        compactness_score = cluster_optimal_resolution(
            adata_int,
            label_key=label_key,
            cluster_key=cluster_key,
            metric=compactness,
        )
    else:
        compactness_score = np.nan

    if kl_divergence_:
        print("KL divergence...")
        kl_div_score = kl_divergence(adata_int, batch_key, embed)
    else:
        kl_div_score = np.nan

    if homogeneity_:
        print("Homogeneity...")
        homogeneity_score = cluster_optimal_resolution(
            adata_int,
            label_key,
            cluster_key,
            metric=homogeneity,
        )
    else:
        homogeneity_score = np.nan

    if fowlkes_mallows_:
        print("Fowlkes Mallows...")
        fm_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=fowlkes_mallows
        )
    else:
        fm_score = np.nan

    if jaccard_:
        print("Jaccard index...")
        jaccard_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=jaccard_index
        )
    else:
        jaccard_score = np.nan

    if matthews_corrcoef_:
        print("Matthews correlation coefficient")
        mcc_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=matthews_corrcoef
        )
    else:
        mcc_score = np.nan

    if modularity_:
        print("Modularity...")
        modularity_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=modularity
        )
    else:
        modularity_score = np.nan

    results = {
        "NMI_cluster/label": nmi_score,
        "ARI_cluster/label": ari_score,
        "silhouette": asw_label,
        "ASW_batch": asw_batch,
        "PCR_batch": pcr_score,
        "cell_cycle_conservation": cc_score,
        "isolated_label_F1": il_score_f1,
        "isolated_label_silhouette": il_score_asw,
        "graph_connectivity": graph_conn_score,
        "kBET": kbet_score,
        "iLISI": ilisi,
        "cLISI": clisi,
        "hvg_overlap": hvg_score,
        "trajectory": trajectory_score,
        "alignment": alignment_score,
        "agreement": agreement_score,
        "scGraph_avg": scgraph_avg,
        "scGraph_rank_pca": scgraph_rank_pca,
        "scGraph_corr_pca": scgraph_corr_pca,
        "scGraph_corr_weighted": scgraph_corr_weighted,
        "ldf_diff": ldf_diff_score,
        "cell_specific_mixing_score": cell_specific_mixing_score_result,
        "crossbatch_knn": crossbatch_knn_score,
        "crossbatch_logistic": crossbatch_logistic_score,
        "batch_ari": batch_ari_score,
        "batch_purity_score": batch_purity,
        "batch_shannon_entropy": batch_shannon_entropy_score,
        "balanced_ARI": bARI_score,
        "balanced AMI": bAMI_score,
        "balanced V-measure": bv_measure_score,
        "batch probing score RF": bps_rf,
        "batch_probing score LR": bps_lr,
        "calinski_harabasz": ch_score,
        "clustering_accuracy": clustacc_score,
        "crossfold_knn": crossfold_knn_score,
        "crossfold_logistic": crossfold_logistic_score,
        "f1_score": f1sm,
        "compactness": compactness_score,
        "KL_divergence": kl_div_score,
        "homogeneity": homogeneity_score,
        "fowlkes_mallows": fm_score,
        "jaccard_index": jaccard_score,
        "matthews_corrcoef": mcc_score,
        "modularity_score": modularity_score,
    }

    return pd.DataFrame.from_dict(results, orient="index")

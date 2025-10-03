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
from .silhouette import silhouette, silhouette_batch
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
    embed,
    embed_pre="X_pca",
    cluster_key="cluster",
    organism="human",
    verbose=False,
    adgbd_=False,
    adjusted_wallace_score_=False,
    agreement_=False,
    alignment_=False,
    adjusted_mutual_info_=False,
    ari_=False,
    ari_fixed_=False,
    bAMI_=False,
    bARI_=False,
    batch_ari_=False,
    batch_probing_score_=False,
    batch_purity_=False,
    bras_=False,
    bv_measure_=False,
    calinski_harabasz_=False,
    cell_cycle_=False,
    cell_specific_mixing_score_=False,
    cluster_batch_entropy_=False,
    clustering_accuracy_=False,
    compactness_=False,
    completeness_=False,
    crossbatch_knn_=False,
    crossbatch_logistic_=False,
    crossfold_knn_=False,
    crossfold_logistic_=False,
    davies_bouldin_=False,
    f1_score_macro_=False,
    fowlkes_mallows_=False,
    graph_conn_=False,
    hvg_score_=False,
    homogeneity_=False,
    isolated_labels_=False,
    jaccard_=False,
    kBET_=False,
    kl_divergence_=False,
    knn_divergence_=False,
    ldf_diff_=False,
    clisi_=False,
    ilisi_=False,
    matthews_corrcoef_=False,
    mixing_=False,
    modularity_=False,
    morans_i_=False,
    nmi_=False,
    nmi_fixed_=False,
    over_correction_=False,
    pcr_=False,
    pptpc_=False,
    purity_=False,
    retained_knn_=False,
    scgraph_=False,
    scaled_variance_=False,
    separation_=False,
    batch_shannon_entropy_=False,
    celltype_shannon_entropy_=False,
    shannon_entropy_nn_=False,
    silhouette_=False,
    trajectory_=False,
    v_measure_=False,
    variation_of_information_=False,
    **kwargs,
):
    """
    Run the metrics specified.

    **kwargs are the keyword arguments for each specific metric.
    """
    check_adata(adata)
    check_batch(batch_key, adata.obs)
    check_batch(label_key, adata.obs)

    check_adata(adata_int)
    check_batch(batch_key, adata_int.obs)
    check_batch(label_key, adata_int.obs)

    # Run `sc.pp.neighbors` to get the kNN graph
    # sc.pp.neighbors(adata, use_rep=embed_pre)
    # sc.pp.neighbors(adata_int, use_rep=embed)

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
            adata_int,
            label_key=label_key,
            embed=embed,
            metric=kwargs.get("si_metric", "euclidean"),
        )
        asw_batch = silhouette_batch(
            adata_int,
            batch_key=batch_key,
            label_key=label_key,
            embed=embed,
            metric=kwargs.get("si_metric", "euclidean"),
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

    if isolated_labels_:
        print("Isolated labels F1...")
        il_score_f1 = isolated_labels(
            adata_int,
            label_key=label_key,
            batch_key=batch_key,
            embed=embed,
            cluster=True,
            iso_threshold=kwargs.get("n_isolated", None),
            verbose=False,
        )
    else:
        il_score_f1 = np.nan

    if isolated_labels_:
        print("Isolated labels ASW...")
        il_score_asw = (
            isolated_labels(
                adata_int,
                label_key=label_key,
                batch_key=batch_key,
                embed=embed,
                cluster=False,
                iso_threshold=kwargs.get("n_isolated", None),
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

    if scgraph_:
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
            type_=kwargs.get("type_", "knn"),
            embed=embed,
            scaled=True,
            verbose=verbose,
        )
    else:
        kbet_score = np.nan

    if clisi_:
        print("cLISI score...")
        clisi = clisi_graph(
            adata_int,
            batch_key=batch_key,
            label_key=label_key,
            type_=kwargs.get("type_", "knn"),
            subsample=kwargs.get("subsample", 0.5) * 100,
            scale=True,
            n_cores=kwargs.get("n_cores", 1),
            verbose=verbose,
        )
    else:
        clisi = np.nan

    if ilisi_:
        print("iLISI score...")
        ilisi = ilisi_graph(
            adata_int,
            batch_key=batch_key,
            type_=kwargs.get("type_", "knn"),
            subsample=kwargs.get("subsample", 0.5) * 100,
            scale=True,
            n_cores=kwargs.get("n_cores", 1),
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
            adata_pre=adata,
            adata_post=adata_int,
            batch_key=batch_key,
            embed_pre=embed_pre,
            embed_post=embed
        )
    else:
        ldf_diff_score = np.nan

    if cell_specific_mixing_score_:
        print("Cell-specific mixing score...")
        cell_specific_mixing_score_result = cell_specific_mixing_score(
            adata_int,
            batch_key=batch_key,
            embed=embed
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
        crossbatch_knn_score = crossbatch_knn(adata_int, batch_key, label_key, embed)
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

    if celltype_shannon_entropy_:
        print("Cell type Shannon entropy...")
        celltype_shannon_entropy_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=celltype_shannon_entropy
        )
    else:
        celltype_shannon_entropy_score = np.nan

    if shannon_entropy_nn_:
        print("Shannon entropy nearest neighbor graph...")
        shannon_entropy_nn_score = shannon_entropy_nn(adata_int, batch_key, label_key)
    else:
        shannon_entropy_nn_score = np.nan

    if ari_fixed_:
        print("ARI fixed...")
        ari_fixed_score = ari_fixed(adata_int, label_key, cluster_key, embed=embed)
    else:
        ari_fixed_score = np.nan

    if bARI_:
        print("Balanced ARI...")
        _, bARI_score = cluster_optimal_resolution(
            adata_int, label_key=label_key, cluster_key=cluster_key, metric=bARI
        )
    else:
        bARI_score = np.nan

    if bAMI_:
        print("Balanced AMI...")
        _, bAMI_score = cluster_optimal_resolution(
            adata_int, label_key=label_key, cluster_key=cluster_key, metric=bAMI
        )
    else:
        bAMI_score = np.nan

    if bv_measure_:
        print("Balanced v-measure...")
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

    if adgbd_:
        print("Average deviation from global batch distribution...")
        adgbd_score = adgbd(adata_int, batch_key)
    else:
        adgbd_score = np.nan

    if adjusted_wallace_score_:
        print("Adjusted Wallace score...")
        aw_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=adjusted_wallace_score
        )
    else:
        aw_score = np.nan

    if adjusted_mutual_info_:
        print("Adjusted mutual information")
        ami_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=ami
        )
    else:
        ami_score = np.nan

    if bras_:
        print("BRAS...")
        bras_score = bras(adata_int, label_key, batch_key, embed)
    else:
        bras_score = np.nan

    if cluster_batch_entropy_:
        print("Cluster batch entropy...")
        cluster_batch_entropy_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=cluster_batch_entropy
        )
    else:
        cluster_batch_entropy_score = np.nan

    if completeness_:
        print("Completeness...")
        completeness_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=completeness
        )
    else:
        completeness_score = np.nan

    if davies_bouldin_:
        print("Davies-Bouldin...")
        davies_bouldin_score = davies_bouldin(adata_int, label_key, embed)
    else:
        davies_bouldin_score = np.nan

    if knn_divergence_:
        print("KNN divergence...")
        knn_div_score = knn_divergence(adata_int, label_key, batch_key, embed)
    else:
        knn_div_score = np.nan
    
    if mixing_:
        print("Mixing score...")
        mixing_score = mixing(adata_int, batch_key, embed)
    else:
        mixing_score = np.nan

    if morans_i_:
        print("Moran's I...")
        moransi_score = morans_i(adata, adata_int, label_key, batch_key, "X_pca", embed)
    else:
        moransi_score = np.nan

    if nmi_fixed_:
        print("NMI fixed...")
        nmi_fixed_score = nmi_fixed(adata_int, label_key, cluster_key)
    else:
        nmi_fixed_score = np.nan

    if over_correction_:
        print("Over correction...")
        oc_score = over_correction(adata_int, label_key)
    else:
        oc_score = np.nan

    if pptpc_:
        print("Positive of positive and true positive cells...")
        pptpc_score = pptpc(adata_int, batch_key, label_key, embed=embed)
    else:
        pptpc_score = np.nan

    if purity_:
        print("Purity...")
        purity_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=purity
        )
    else:
        purity_score = np.nan

    if retained_knn_:
        print("Retained KNN...")
        retained_knn_score = retained_knn(adata, adata_int, batch_key, embed)
    else:
        retained_knn_score = np.nan

    if scaled_variance_:
        print("Scaled variance...")
        sv_score = scaled_variance(adata_int, batch_key, embed)
    else:
        sv_score = np.nan

    if separation_:
        separation_score = cluster_optimal_resolution(
            adata_int,
            label_key,
            cluster_key,
            metric=separation,
            kwargs={"embed": embed},
        )
    else:
        separation_score = np.nan

    if v_measure_:
        print("V measure...")
        v_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=v_measure
        )
    else:
        v_score = np.nan

    if variation_of_information_:
        print("Variation of information...")
        vi_score = cluster_optimal_resolution(
            adata_int, label_key, cluster_key, metric=variation_of_information
        )
    else:
        vi_score = np.nan

    results = {
        "abgbd": adgbd_score,
        "adjusted_wallace_score": aw_score,
        "alignment": alignment_score,
        "agreement": agreement_score,
        "AMI": ami_score,
        "ARI_cluster/label": ari_score,
        "ARI_fixed": ari_fixed_score,
        "ASW_batch": asw_batch,
        "balanced_AMI": bAMI_score,
        "balanced_V_measure": bv_measure_score,
        "balanced_ARI": bARI_score,
        "batch_robing_score_RF": bps_rf,
        "batch_probing_score_LR": bps_lr,
        "batch_ari": batch_ari_score,
        "batch_purity_score": batch_purity,
        "batch_shannon_entropy": batch_shannon_entropy_score,
        "BRAS": bras_score,
        "calinski_harabasz": ch_score,
        "cell_cycle_conservation": cc_score,
        "cell_specific_mixing_score": cell_specific_mixing_score_result,
        "cell_type_shannon_entropy": celltype_shannon_entropy_score,
        "cLISI": clisi,
        "clustering_accuracy": clustacc_score,
        "cluster_batch_entropy": cluster_batch_entropy_score,
        "compactness": compactness_score,
        "completeness": completeness_score,
        "crossbatch_knn": crossbatch_knn_score,
        "crossbatch_logistic": crossbatch_logistic_score,
        "crossfold_knn": crossfold_knn_score,
        "crossfold_logistic": crossfold_logistic_score,
        "davies_bouldin": davies_bouldin_score,
        "f1_score": f1sm,
        "fowlkes_mallows": fm_score,
        "graph_connectivity": graph_conn_score,
        "homogeneity": homogeneity_score,
        "hvg_overlap": hvg_score,
        "iLISI": ilisi,
        "isolated_label_F1": il_score_f1,
        "isolated_label_silhouette": il_score_asw,
        "jaccard_index": jaccard_score,
        "kBET": kbet_score,
        "KL_divergence": kl_div_score,
        "knn_divergence": knn_div_score,
        "ldf_diff": ldf_diff_score,
        "NMI_cluster/label": nmi_score,
        "matthews_corrcoef": mcc_score,
        "mixing": mixing_score,
        "modularity_score": modularity_score,
        "morans_I": moransi_score,
        "nmi_fixed": nmi_fixed_score,
        "over_correction": oc_score,
        "pcr_batch": pcr_score,
        "pptpc": pptpc_score,
        "purity": purity_score,
        "retained_knn": retained_knn_score,
        "scaled_variance": sv_score,
        "scGraph_avg": scgraph_avg,
        "scGraph_corr_pca": scgraph_corr_pca,
        "scGraph_corr_weighted": scgraph_corr_weighted,
        "scGraph_rank_pca": scgraph_rank_pca,
        "separation": separation_score,
        "shannon_entropy_nn": shannon_entropy_nn_score,
        "silhouette": asw_label,
        "trajectory": trajectory_score,
        "v_measure": v_score,
        "variation_of_information": vi_score,
    }

    return pd.DataFrame.from_dict(results, orient="index")

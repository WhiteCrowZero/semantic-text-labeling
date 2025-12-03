import os
import json

import numpy as np
import pandas as pd
import hdbscan
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from src.common.log_utils import init_logger
from src.config.settings import LOG_DIR, CONFIG

logger = init_logger(
    name="clustering",
    module_name=__name__,
    log_dir=os.path.join(LOG_DIR, "model"),
)


# ========== 指标计算 ==========

def evaluate_internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    内部指标评估（只看 labels != -1 的样本）：
      - Silhouette（越大越好）
      - Calinski-Harabasz（越大越好）
      - Davies-Bouldin（越小越好）
    """
    mask = labels != -1
    X_in = X[mask]
    y_in = labels[mask]

    metrics = {
        "n_samples_total": int(X.shape[0]),
        "n_samples_in_clusters": int(mask.sum()),
        "n_clusters": 0,
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
    }

    if X_in.shape[0] == 0:
        logger.warning("[Clustering] 全是噪声，无法计算内部指标")
        return metrics

    n_clusters = len(set(y_in))
    metrics["n_clusters"] = int(n_clusters)

    if n_clusters < 2 or X_in.shape[0] <= n_clusters:
        logger.warning(
            f"[Clustering] 有效簇数={n_clusters}，样本数={X_in.shape[0]}，跳过内部指标计算"
        )
        return metrics

    metrics["silhouette"] = float(silhouette_score(X_in, y_in))
    metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_in, y_in))
    metrics["davies_bouldin"] = float(davies_bouldin_score(X_in, y_in))

    logger.info(
        f"[Clustering] 指标: silhouette={metrics['silhouette']:.4f}, "
        f"calinski_harabasz={metrics['calinski_harabasz']:.2f}, "
        f"davies_bouldin={metrics['davies_bouldin']:.4f}"
    )
    return metrics


# ========== 降维 & 可视化 ==========

def pca_reduce(embeddings: np.ndarray, n_components: int = 50) -> np.ndarray:
    """
    PCA 降维：
      - 降低维度，减轻聚类压力
      - 让距离/密度更稳定
    """
    logger.info(f"[Clustering] PCA 降维到 {n_components} 维")
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(embeddings)
    logger.info(
        f"[Clustering] PCA 解释方差占比: {pca.explained_variance_ratio_.sum():.4f}"
    )
    return X_reduced


def plot_clusters_2d(
    X: np.ndarray,
    labels: np.ndarray,
    fig_path: str,
    title: str = "Clusters (2D PCA)",
    max_points: int = 5000,
):
    """
    用 PCA 降到 2 维做散点图，主要看整体结构。
    默认用 KMeans 的标签做展示。
    """
    n_samples = X.shape[0]
    if n_samples > max_points:
        idx = np.random.choice(n_samples, size=max_points, replace=False)
        X = X[idx]
        labels = labels[idx]

    logger.info("[Clustering] 生成 2D 聚类可视化图")
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X)

    x = X_2d[:, 0]
    y = X_2d[:, 1]

    plt.figure(figsize=(10, 8))
    unique_labels = sorted(set(labels))

    for lab in unique_labels:
        mask = labels == lab
        if lab == -1:
            plt.scatter(x[mask], y[mask], s=5, c="lightgray", alpha=0.5, label="noise")
        else:
            plt.scatter(x[mask], y[mask], s=5, alpha=0.7, label=f"cluster {lab}")

    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=200)
    plt.close()

    logger.info(f"[Clustering] 聚类可视化已保存到: {fig_path}")


# ========== 主流程：情感切分 + KMeans + HDBSCAN(中心) ==========

def run_clustering_pipeline(
    npz_path: str,
    meta_csv: str,
    output_csv: str,
    output_fig_path: str = None,
    output_metrics_path: str = None,
):
    """
    聚类策略：
      1. embeddings 假定为 mean pooling 结果。
      2. 按情感标签切分（label_column）。
      3. 每个情感块：
           - KMeans 在样本级聚类，得到 cluster_kmeans。
           - 取 KMeans 簇中心，再用 HDBSCAN 在“簇中心空间”上聚类，
             得到对簇的二次聚合/噪声识别，再映射回样本上，形成 cluster_hdbscan。
      4. 输出：
           - reviews_with_cluster.csv：cluster_kmeans / cluster_hdbscan 两列。
           - hdbscan_metrics.json：KMeans/HDBSCAN 各自的内部指标。
           - hdbscan_clusters.png：KMeans 的 2D 可视化。
    """
    logger.info(f"[Clustering] 加载向量: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]  # (N, hidden)

    logger.info(f"[Clustering] 向量形状: {embeddings.shape}")

    # 1) PCA 降维
    X = pca_reduce(embeddings, n_components=50)

    # 2) 加载 meta & 情感标签
    logger.info(f"[Clustering] 加载元信息: {meta_csv}")
    df_meta = pd.read_csv(meta_csv)

    if len(df_meta) != X.shape[0]:
        raise ValueError(
            f"meta 行数({len(df_meta)}) != 向量数({X.shape[0]})，请检查数据流程"
        )

    sentiment_col = CONFIG["EMBEDDING"]["label_column"]  # e.g. 'sentiment_value'
    if sentiment_col not in df_meta.columns:
        raise ValueError(
            f"meta 中找不到情感列 '{sentiment_col}'，现有列: {df_meta.columns.tolist()}"
        )

    sentiments = df_meta[sentiment_col].to_numpy()
    unique_sentiments = sorted(pd.unique(sentiments))
    logger.info(f"[Clustering] 情感取值: {unique_sentiments}")

    # 全局标签
    n_samples = X.shape[0]
    kmeans_labels_global = np.full(n_samples, -1, dtype=int)
    hdbscan_labels_global = np.full(n_samples, -1, dtype=int)

    # 让不同情感的簇 ID 不冲突
    kmeans_offset = 0
    hdbscan_offset = 0

    base_k = CONFIG.get("CLUSTERING", {}).get("kmeans_n_clusters", 20)

    # 3) 按情感分块
    for s in unique_sentiments:
        indices = np.where(sentiments == s)[0]
        X_s = X[indices]
        n_s = X_s.shape[0]

        logger.info(f"[Clustering] 处理情感={s}，样本数={n_s}")

        if n_s < 5:
            logger.warning(
                f"[Clustering] 情感={s} 样本过少(<5)，跳过聚类，保持 -1 标签"
            )
            continue

        # ---------- 3.1 KMeans：样本级 ----------
        k = min(base_k, n_s)
        logger.info(
            f"[Clustering] 情感={s}，KMeans 聚类，n_clusters={k}"
        )
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=min(1024, n_s),
            random_state=42,
        )
        local_kmeans = kmeans.fit_predict(X_s)           # [n_s]
        centers = kmeans.cluster_centers_                # [k, dim]

        # 记录全局 KMeans 标签
        kmeans_labels_global[indices] = local_kmeans + kmeans_offset
        kmeans_offset += centers.shape[0]

        # ---------- 3.2 HDBSCAN：簇中心级 ----------
        logger.info(f"[Clustering] 情感={s}，对 KMeans 簇中心执行 HDBSCAN 聚类")

        # 根据簇数量给一个相对宽松的密度门槛
        k_centers = centers.shape[0]
        if k_centers < 3:
            logger.warning(
                f"[Clustering] 情感={s} KMeans 簇数过少({k_centers})，跳过 HDBSCAN"
            )
            continue

        min_cluster_size = max(2, k_centers // 5)  # 比如 k=20 => 4
        min_samples = 1

        h_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            core_dist_n_jobs=-1,
        )
        center_labels = h_clusterer.fit_predict(centers)   # [k]

        n_center_clusters = len(set(center_labels) - {-1})
        n_center_noise = int((center_labels == -1).sum())
        logger.info(
            f"[Clustering] 情感={s}，中心级 HDBSCAN 簇数={n_center_clusters}，"
            f"噪声中心数={n_center_noise}"
        )

        if n_center_clusters == 0:
            # 全部中心都是噪声，则该情感块的样本在 HDBSCAN 视角下全部保留为 -1
            logger.warning(
                f"[Clustering] 情感={s}，中心全为噪声，样本级 HDBSCAN 标签保持 -1"
            )
            continue

        # 将中心级标签映射到样本级：
        # 每个样本的 HDBSCAN label = 它所属的 KMeans 簇中心的 label
        local_hdb = center_labels[local_kmeans]       # [n_s]

        # 非噪声中心映射到全局 id
        mask_non_noise = local_hdb != -1
        if mask_non_noise.any():
            # 中心非噪声标签的最大值
            max_center_label = center_labels[center_labels != -1].max()
            hdbscan_labels_global[indices[mask_non_noise]] = \
                local_hdb[mask_non_noise] + hdbscan_offset
            hdbscan_offset += max_center_label + 1

    # 4) 全局内部指标
    logger.info("[Clustering] 计算 KMeans 全局指标")
    kmeans_metrics = evaluate_internal_metrics(X, kmeans_labels_global)

    logger.info("[Clustering] 计算 HDBSCAN 全局指标")
    hdbscan_metrics = evaluate_internal_metrics(X, hdbscan_labels_global)

    metrics_all = {
        "kmeans": kmeans_metrics,
        "hdbscan": hdbscan_metrics,
    }

    # 5) 输出结果
    df_meta["cluster_kmeans"] = kmeans_labels_global
    df_meta["cluster_hdbscan"] = hdbscan_labels_global
    df_meta.to_csv(output_csv, index=False)
    logger.info(f"[Clustering] 聚类结果已保存到: {output_csv}")

    # 6) 指标 JSON
    if output_metrics_path:
        os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
        with open(output_metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_all, f, ensure_ascii=False, indent=2)
        logger.info(f"[Clustering] 聚类指标已保存到: {output_metrics_path}")

    # 7) 可视化：用 KMeans 结果做一张 2D 总图
    if output_fig_path:
        plot_clusters_2d(
            X,
            kmeans_labels_global,
            output_fig_path,
            title="KMeans Clusters by Sentiment (2D PCA)",
        )

    return df_meta, kmeans_labels_global, hdbscan_labels_global, metrics_all


if __name__ == "__main__":
    clustering_cfg = CONFIG.get("CLUSTERING", {})

    npz_path = clustering_cfg.get("input_npz")
    meta_csv = clustering_cfg.get("input_meta_csv")
    output_csv = clustering_cfg.get("output_cluster_csv")
    output_fig_path = clustering_cfg.get("output_fig_path")
    output_metrics_path = clustering_cfg.get("output_metrics_path")

    run_clustering_pipeline(
        npz_path=npz_path,
        meta_csv=meta_csv,
        output_csv=output_csv,
        output_fig_path=output_fig_path,
        output_metrics_path=output_metrics_path,
    )

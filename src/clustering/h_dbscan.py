import os
import time

import numpy as np
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
from src.embeddings.bert import BertEmbedder


class LabelClusterer:
    """
    用于对大模型生成的标签（字符串）做聚类：
      - 输入：List[str] (标签)
      - 输出：KMeans | HDBSCAN 聚类编号
      - 可视化输出（可选）
    """

    def __init__(
        self,
        embedder: BertEmbedder,
    ):
        self.embedder = embedder  # 你自己的 BERT 向量模型
        self.config = CONFIG.get("CLUSTERING", {})
        self.n_components = self.config.get("PCA_COMPONENTS", 50)
        self.kmeans_k = self.config.get("KMEANS_K", 20)
        self.fig_dir = self.config.get("FIG_DIR", None)
        self.default_algo = self.config.get("DEFAULT_ALGO", "kmeans")  # 默认算法

        self.logger = init_logger(
            name="clustering",
            module_name=str(self.__class__.__name__),
            log_dir=os.path.join(LOG_DIR, "model"),
        )

    # --------- PCA 降维 ---------
    def pca_reduce(self, embeddings: np.ndarray) -> np.ndarray:
        n_samples, n_features = embeddings.shape
        n_components = min(self.n_components, n_samples)
        self.logger.info(f"[PCA] 降维到 {n_components} 维")
        pca = PCA(n_components=n_components, random_state=42)
        X = pca.fit_transform(embeddings)
        self.logger.info(f"[PCA] 总解释方差: {pca.explained_variance_ratio_.sum():.4f}")
        return X

    # --------- 内部指标 ---------
    def evaluate_metrics(self, X: np.ndarray, labels: np.ndarray) -> dict:
        mask = labels != -1
        X_in, y_in = X[mask], labels[mask]

        metrics = {
            "n_samples": int(X.shape[0]),
            "n_in_clusters": int(mask.sum()),
            "n_clusters": len(set(y_in)) if y_in.size > 0 else 0,
            "silhouette": None,
            "calinski": None,
            "davies_bouldin": None,
        }

        if y_in.size == 0 or metrics["n_clusters"] < 2:
            return metrics

        metrics["silhouette"] = float(silhouette_score(X_in, y_in))
        metrics["calinski"] = float(calinski_harabasz_score(X_in, y_in))
        metrics["davies_bouldin"] = float(davies_bouldin_score(X_in, y_in))

        return metrics

    # --------- 2D 可视化 ---------
    def plot_clusters(self, X: np.ndarray, labels: np.ndarray, algo_name: str) -> str:
        if not self.fig_dir:
            self.logger.error("[Plot] 未设置 fig_dir")
            return ""

        os.makedirs(self.fig_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(self.fig_dir, f"{algo_name}_clusters_{timestamp}.png")

        p2 = PCA(n_components=2, random_state=42)
        X2 = p2.fit_transform(X)

        plt.figure(figsize=(10, 8))
        for lab in sorted(set(labels)):
            mask = labels == lab
            if lab == -1:
                plt.scatter(
                    X2[mask, 0], X2[mask, 1], s=5, c="gray", alpha=0.5, label="noise"
                )
            else:
                plt.scatter(
                    X2[mask, 0], X2[mask, 1], s=5, alpha=0.7, label=f"cluster {lab}"
                )
        plt.legend(fontsize=8)
        plt.title(f"{algo_name.upper()} Label Clusters (2D PCA)")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()

        self.logger.info(f"[Plot] 聚类图保存到: {fig_path}")
        return fig_path

    # --------- 主流程 ---------
    def fit_predict(
        self,
        labels: list[str],
        algo: str = None,
        save_metrics: bool = True,
        visualize: bool = True,
    ):
        """
        输入：
          - labels: list[str]
          - algo: 'kmeans' | 'hdbscan' | None（默认 self.default_algo）
          - save_metrics: 是否保存指标 JSON
          - visualize: 是否生成可视化
        输出：
          - dict 包含聚类结果
        """
        algo = (algo or self.default_algo).lower()
        self.logger.info(f"[Embed] 编码 {len(labels)} 条标签")
        embeddings = self.embedder.embed_batch(labels)
        X = self.pca_reduce(embeddings)
        n_samples = X.shape[0]

        results = {}
        if algo in ("kmeans", "both"):
            k = min(self.kmeans_k, n_samples)
            self.logger.info(f"[KMeans] 聚类数={k}")
            kmeans = MiniBatchKMeans(
                n_clusters=k, batch_size=min(1024, n_samples), random_state=42
            )
            kmeans_labels = kmeans.fit_predict(X)
            kmeans_fig_path = (
                self.plot_clusters(X, kmeans_labels, "kmeans") if visualize else None
            )
            results["kmeans"] = {
                "labels": kmeans_labels,
                "metrics": self.evaluate_metrics(X, kmeans_labels),
                "fig_path": kmeans_fig_path,
            }

        if algo in ("hdbscan", "both"):
            # 如果没有 KMeans，HDBSCAN 在 embeddings 上直接聚类
            if "kmeans" in results:
                centers = (
                    MiniBatchKMeans(
                        n_clusters=min(self.kmeans_k, n_samples), random_state=42
                    )
                    .fit(X)
                    .cluster_centers_
                )
            else:
                centers = X

            if centers.shape[0] >= 3:
                min_cluster = max(2, centers.shape[0] // 5)
                hdb = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster,
                    min_samples=1,
                    metric="euclidean",
                    core_dist_n_jobs=-1,
                )
                hdb_center_labels = hdb.fit_predict(centers)
                if "kmeans" in results:
                    hdb_labels = hdb_center_labels[results["kmeans"]["labels"]]
                else:
                    hdb_labels = hdb_center_labels
                hdbscan_fig_path = (
                    self.plot_clusters(X, hdb_labels, "hdbscan") if visualize else None
                )
                results["hdbscan"] = {
                    "labels": hdb_labels,
                    "metrics": self.evaluate_metrics(X, hdb_labels),
                    "fig_path": hdbscan_fig_path,
                }

            else:
                self.logger.warning("[HDBSCAN] 样本或簇中心不足，跳过 HDBSCAN")
                results["hdbscan"] = {
                    "labels": np.full(n_samples, -1, dtype=int),
                    "metrics": {},
                    "fig_path": "",
                }

        return results


if __name__ == "__main__":
    embedder = BertEmbedder()
    clusterer = LabelClusterer(embedder)

    labels = [
        "手机电池续航",
        "手机屏幕质量",
        "相机性能",
        "笔记本键盘舒适度",
        "电视画质",
        "手机软件更新",
        "耳机音质",
        "冰箱能效",
        "笔记本性能",
        "手表健康追踪",
        "手机屏幕质量",
        "手机电池续航",
        "相机清晰度",
        "笔记本显示效果",
        "电视音质",
        "手机流畅度",
        "耳机舒适度",
        "冰箱温控效果",
        "笔记本电池续航",
        "手表智能提醒",
        "手机快充功能",
        "手机摄像头清晰度",
        "相机操作体验",
        "笔记本性能稳定性",
        "电视色彩还原",
        "手机售后服务",
        "耳机蓝牙稳定性",
        "冰箱噪音控制",
        "笔记本屏幕显示效果",
        "手表防水性能",
        "手机软件更新",
        "手机屏幕耐用性",
        "相机对焦速度",
        "笔记本触控板",
        "电视安装便捷性",
        "手机硬件配置",
        "耳机防水性能",
        "冰箱存储空间",
        "笔记本性价比",
        "手表续航时间",
        "手机操作系统流畅性",
        "手机边框设计",
        "相机低光拍摄表现",
        "笔记本散热效果",
        "电视智能功能",
        "手机摄像头拍照效果",
        "耳机配对速度",
        "冰箱省电模式",
        "笔记本散热设计",
        "手表心率监测",
    ]

    # --------- 聚类 ---------
    results = clusterer.fit_predict(labels, visualize=True, save_metrics=True)

    # --------- 输出结果 ---------
    print("Labels:", labels)
    print("KMeans labels:", results["kmeans"]["labels"])
    print("HDBSCAN labels:", results["hdbscan"]["labels"])
    print("Metrics:", results)

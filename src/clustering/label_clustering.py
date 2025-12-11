import os
import time

import numpy as np
import hdbscan
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
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

    def __init__(self):
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

        X = np.asarray(X)
        labels = np.asarray(labels)

        n_samples = X.shape[0]
        if labels.shape[0] != n_samples:
            self.logger.error(
                f"[Plot] labels 长度 {labels.shape[0]} != 样本数 {n_samples}，跳过绘图"
            )
            return ""

        if n_samples < 2:
            self.logger.warning("[Plot] 样本太少，跳过 t-SNE 绘图")
            return ""

        os.makedirs(self.fig_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(
            self.fig_dir, f"{algo_name}_tsne_clusters_{timestamp}.png"
        )

        # -------- t-SNE 降维 --------
        # perplexity 必须 < n_samples，这里做一个安全设置
        perp = min(30, max(2, n_samples // 3))
        if perp >= n_samples:
            perp = max(1, n_samples - 1)

        self.logger.info(
            f"[Plot] 使用 t-SNE 可视化，perplexity={perp}, n_samples={n_samples}"
        )

        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            learning_rate="auto",
            init="random",
            random_state=42,
        )
        X2 = tsne.fit_transform(X)  # (n_samples, 2)

        # -------- 画图 --------
        plt.figure(figsize=(10, 8))
        # 颜色序列：可以自己换成喜欢的；数量不够会循环使用
        color_cycle = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]

        # 1. 先画噪声（-1）
        mask_noise = labels == -1
        if np.any(mask_noise):
            plt.scatter(
                X2[mask_noise, 0],
                X2[mask_noise, 1],
                s=8,
                c="lightgray",
                alpha=0.5,
                label="noise",
            )

        # 2. 其他标签按“顺序列表”+颜色序列来画
        #    这里用 sorted 保证顺序稳定，你也可以换成自己控制的顺序列表
        ordered_labels = [lab for lab in sorted(set(labels)) if lab != -1]

        for idx, lab in enumerate(ordered_labels):
            mask = labels == lab
            if not np.any(mask):
                continue

            color = color_cycle[idx % len(color_cycle)]  # 顺序映射到颜色
            plt.scatter(
                X2[mask, 0],
                X2[mask, 1],
                s=8,
                alpha=0.7,
                color=color,
                label=f"cluster {lab}",
            )

        plt.legend(fontsize=8)
        plt.title(f"{algo_name.upper()} Label Clusters (t-SNE 2D)")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()

        self.logger.info(f"[Plot] 聚类 t-SNE 图保存到: {fig_path}")
        return fig_path

    # --------- 主流程 ---------
    def fit_predict(
        self,
        embeddings: np.ndarray,
        algo: str = None,
        kmeans_k: int = None,
        visualize: bool = True,
    ):
        """
        输入：
          - embeddings: np.ndarray, 形状 (n_samples, dim)
          - algo: 'kmeans' | 'hdbscan' | 'both' | None（默认 self.default_algo）
          - visualize: 是否生成 t-SNE 可视化
        输出：
          - dict 包含聚类结果
        """
        algo = (algo or self.default_algo).lower()
        X = np.asarray(embeddings)
        n_samples = X.shape[0]

        results = {}

        # ---------- KMeans ----------
        if algo in ("kmeans", "both"):
            # 真实使用的 k
            if kmeans_k is not None:
                k = kmeans_k
            else:
                k = self.kmeans_k

            k = int(min(max(2, k), n_samples))  # 至少2类，不超过样本数

            if k < 2:
                self.logger.warning("[KMeans] 样本太少，跳过 KMeans")
            else:
                self.logger.info(f"[KMeans] 聚类数 = {k}")
                kmeans_model = MiniBatchKMeans(
                    n_clusters=k,
                    batch_size=min(1024, n_samples),
                    random_state=42,
                )
                kmeans_labels = kmeans_model.fit_predict(X)

                kmeans_fig_path = (
                    self.plot_clusters(X, kmeans_labels, "kmeans")
                    if (visualize and n_samples > 1)
                    else None
                )

                results["kmeans"] = {
                    "labels": kmeans_labels,
                    "metrics": self.evaluate_metrics(X, kmeans_labels),
                    "fig_path": kmeans_fig_path,
                }

        # ---------- HDBSCAN ----------
        if algo in ("hdbscan", "both"):
            # 1) 在样本上聚类，不走 KMeans center
            centers = X
            self.logger.info(f"[HDBSCAN] 直接在样本上聚类，样本数 = {centers.shape[0]}")

            n_centers = centers.shape[0]
            if n_centers >= 5:
                # 2) 先归一化 + UMAP 降维到默认维
                centers_norm = normalize(centers)  # L2 归一化
                n_components = min(
                    self.n_components, centers_norm.shape[1], n_centers - 1
                )

                if n_components >= 2:
                    reducer = umap.UMAP(
                        n_components=n_components,
                        n_neighbors=15,  # 可按需调整
                        min_dist=0.1,  # 可按需调整
                        random_state=42,
                    )
                    centers_reduced = reducer.fit_transform(centers_norm)
                else:
                    centers_reduced = centers_norm

                # # 2) 先归一化 + PCA 降维到默认维
                # centers_norm = normalize(centers)  # L2 归一化
                # n_components = min(
                #     self.n_components, centers_norm.shape[1], n_centers - 1
                # )
                # if n_components >= 2:
                #     pca = PCA(n_components=n_components, random_state=42)
                #     centers_reduced = pca.fit_transform(centers_norm)
                # else:
                #     centers_reduced = centers_norm

                # 3) 设置更温和的 min_cluster_size / min_samples
                min_cluster = max(10, int(0.02 * n_centers))
                min_samples = max(5, int(0.01 * n_centers))
                # min_cluster = 20
                # min_samples = 5

                self.logger.info(
                    f"[HDBSCAN] min_cluster_size={min_cluster}, "
                    f"min_samples={min_samples}, n_centers={n_centers}"
                )

                hdb = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster,
                    min_samples=min_samples,
                    metric="euclidean",  # 向量已经归一化 + 降维
                    cluster_selection_method="leaf",
                    core_dist_n_jobs=1,
                )
                hdb_labels = hdb.fit_predict(centers_reduced)  # 直接是样本级标签

                # 看一下 label 分布
                uniq, cnt = np.unique(hdb_labels, return_counts=True)
                self.logger.info(f"[HDBSCAN] label 分布: {dict(zip(uniq, cnt))}")

                hdbscan_fig_path = (
                    self.plot_clusters(centers_reduced, hdb_labels, "hdbscan")
                    if (visualize and n_centers > 1)
                    else None
                )

                results["hdbscan"] = {
                    "labels": hdb_labels,
                    "metrics": self.evaluate_metrics(centers_reduced, hdb_labels),
                    "fig_path": hdbscan_fig_path,
                }
            else:
                self.logger.warning("[HDBSCAN] 样本不足，跳过 HDBSCAN")
                results["hdbscan"] = {
                    "labels": np.full(n_centers, -1, dtype=int),
                    "metrics": {},
                    "fig_path": "",
                }

        return results


if __name__ == "__main__":
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

    embedder = BertEmbedder()
    clusterer = LabelClusterer()

    embeddings = embedder.embed_batch(labels)
    # --------- 聚类 ---------
    results = clusterer.fit_predict(embeddings, visualize=True)
    print(results)

    """
    result:
    {
        "kmeans": {
            "labels": array([16,  4,  7,  8,  0,  5,  3,  6, 11, 12,  4, 16,  7,  2,  0, 10,  3,
                        6, 16, 12, 16,  7,  7, 11,  0,  9, 19, 18,  2, 17,  5,  4, 15,  8,
                        0,  9, 13,  6, 11, 17, 10, 14,  7,  1,  0,  7,  3,  6,  1, 12],
                      dtype=int32),
            "metrics": {
                "n_samples": 50,
                "n_in_clusters": 50,
                "n_clusters": 20,
                "silhouette": 0.22446654736995697,
                "calinski": 4.992252826690674,
                "davies_bouldin": 0.9360142646561812,
            },
            "fig_path": "D:\\ComputerScience\\Python\\temp\\ml_homework3\\data\\output\\figures\\kmeans_clusters_20251203_193230.png",
        },
        "hdbscan": {
            "labels": array([ 2,  2,  2,  0,  0, -1,  1,  0,  0,  1,  2,  2,  2,  0,  0,  2,  1,
                        0,  2,  1,  2,  2,  2,  0,  0,  2,  1,  0,  0,  1, -1,  2,  2,  0,
                        0,  2,  1,  0,  0,  1,  2, -1,  2,  0,  0,  2,  1,  0,  0,  1]),
            "metrics": {
                "n_samples": 50,
                "n_in_clusters": 47,
                "n_clusters": 3,
                "silhouette": 0.12346836179494858,
                "calinski": 5.97938346862793,
                "davies_bouldin": 2.255895295566246,
            },
            "fig_path": "D:\\ComputerScience\\Python\\temp\\ml_homework3\\data\\output\\figures\\hdbscan_clusters_20251203_193231.png",
        },
    }
    """

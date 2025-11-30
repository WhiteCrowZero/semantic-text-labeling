"""
功能：
  - 从 CSV 或 THUCNews 目录 加载数据集
  - 用 BERT 提取句子级 mean pooling 句向量
  - 把向量保存成 .npy / .npz，元信息保存成 .csv

使用方式：
  1. 修改 CONFIG 里的 EMBEDDING 配置
  2. 在 PyCharm 里直接运行这个文件
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel

from src.common.log_utils import init_logger
from src.config.settings import CONFIG, LOG_DIR


class CsvBertEmbedder:
    """
    基于 BERT 的文本编码器：
      - 支持两种数据源：
          - CSV：input_csv + text_column
          - THUCNews 目录：input_dir（子目录为类别名，文件为新闻）
      - 编码 mean pooling 句向量
      - 输出 npy/npz + meta CSV
    """

    def __init__(self, config: dict):
        self.config = config
        self.logger = init_logger(
            name="embeddings",
            module_name=str(self.__class__.__name__),
            log_dir=os.path.join(LOG_DIR, "model"),
        )
        self.model = None
        self.tokenizer = None
        self.device = None

    # ========= 设备 & 模型加载 =========

    def _get_device(self) -> str:
        device = self.config.get("device")
        if device:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def build_model_and_tokenizer(self):
        """
        加载 tokenizer 和 BERT 模型：
        - 若 local_model_path 目录存在有效模型文件 => 从本地加载
        - 否则从 HuggingFace 下载，并保存到 local_model_path
        """
        device = self._get_device()
        model_name = self.config["model_name"]
        local_model_path = self.config.get("local_model_path")

        use_local = False
        if local_model_path:
            config_path = os.path.join(local_model_path, "config.json")
            if os.path.exists(config_path):
                use_local = True

        if use_local:
            self.logger.info(f"[Embedding] 从本地加载模型: {local_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            model = AutoModel.from_pretrained(local_model_path).to(device)
        else:
            self.logger.info(
                f"[Embedding] 本地未找到模型，开始从远程下载: {model_name}"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)

            if local_model_path:
                os.makedirs(local_model_path, exist_ok=True)
                tokenizer.save_pretrained(local_model_path)
                model.save_pretrained(local_model_path)
                self.logger.info(f"[Embedding] 已将模型保存到本地: {local_model_path}")
            else:
                self.logger.warning(
                    "[Embedding] 未配置 local_model_path，模型仅存在于缓存，不会单独落盘目录"
                )

        model.eval()
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

        self.logger.info(f"[Embedding] 模型加载完成，使用设备: {device}")

    # ========= 数据加载：CSV / THUCNews =========

    def _load_texts_from_csv(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        从 CSV 加载文本和元信息（id、label 等）。
        要求 CSV 至少有一列文本列 text_column。
        """
        input_csv = self.config["input_csv"]
        text_column = self.config["text_column"]

        self.logger.info(f"[Embedding] 从 CSV 加载数据集: {input_csv}")
        df = pd.read_csv(input_csv)

        if text_column not in df.columns:
            raise ValueError(
                f"CSV 中找不到文本列 '{text_column}'，现有列: {df.columns.tolist()}"
            )

        texts = df[text_column].astype(str).tolist()
        self.logger.info(f"[Embedding] 样本数: {len(texts)}")
        return df, texts

    def _load_texts_from_thucnews_dir(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        从 THUCNews 目录加载数据：
        目录结构示例：
            data/thucnews/
                体育/
                    123.txt
                    456.txt
                财经/
                    ...
        约定：
          - label_name = 子目录名
          - label = 按 label_name 排序后的索引（0,1,2,...）
          - id = label_name + 文件名
          - content = 文件全文
        """
        input_dir = self.config["input_dir"]
        text_column = self.config.get("text_column", "content")
        id_column = self.config.get("id_column", "id")
        label_column = self.config.get("label_column", "label")

        if not os.path.isdir(input_dir):
            raise ValueError(f"[Embedding] THUCNews 目录不存在: {input_dir}")

        self.logger.info(f"[Embedding] 从 THUCNews 目录加载数据: {input_dir}")

        label_names = [
            d for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
        ]
        label_names = sorted(label_names)
        if not label_names:
            raise ValueError(f"[Embedding] 目录 {input_dir} 下未找到任何子目录（类别）")

        self.logger.info(f"[Embedding] 检测到类别: {label_names}")

        rows = []
        total = 0

        max_samples_per_class = self.config.get("max_samples_per_class")  # 可选截断

        for label_idx, label_name in enumerate(label_names):
            label_dir = os.path.join(input_dir, label_name)
            files = [
                f for f in os.listdir(label_dir)
                if os.path.isfile(os.path.join(label_dir, f)) and f.endswith(".txt")
            ]

            if max_samples_per_class is not None:
                files = files[:max_samples_per_class]

            self.logger.info(
                f"[Embedding] 类别={label_name} (label={label_idx})，文件数={len(files)}"
            )

            for fname in files:
                fpath = os.path.join(label_dir, fname)
                # 防止编码问题，ignore 直接跳过非法字节
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                if not content:
                    continue

                row_id = f"{label_name}_{fname}"
                rows.append(
                    {
                        id_column: row_id,
                        label_column: label_idx,
                        "label_name": label_name,
                        text_column: content,
                    }
                )
                total += 1

        if not rows:
            raise ValueError(f"[Embedding] 未从 {input_dir} 读取到任何文本，请检查目录结构和编码")

        df = pd.DataFrame(rows)
        texts = df[text_column].astype(str).tolist()
        self.logger.info(f"[Embedding] THUCNews 加载完成，样本数: {total}")
        return df, texts

    def load_texts(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        根据 dataset_type 选择加载模式：
          - csv
          - thucnews_dir
        """
        dataset_type = self.config.get("dataset_type", "csv")
        if dataset_type == "csv":
            return self._load_texts_from_csv()
        elif dataset_type == "thucnews_dir":
            return self._load_texts_from_thucnews_dir()
        else:
            raise ValueError(f"不支持的数据集类型 dataset_type={dataset_type}")

    # ========= 编码 =========

    def encode_batch(self, sentences: List[str]) -> np.ndarray:
        """
        对一批句子做 BERT 编码，返回 mean pooling 向量（已 L2 归一化）。
        """
        if self.model is None or self.tokenizer is None or self.device is None:
            raise RuntimeError("模型未初始化，请先调用 build_model_and_tokenizer().")

        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.config["max_length"],
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            last_hidden = outputs.last_hidden_state           # [batch, seq_len, hidden]
            attention_mask = encoded["attention_mask"]        # [batch, seq_len]

            # mean pooling
            mask = attention_mask.unsqueeze(-1)               # [batch, seq_len, 1]
            sum_embeddings = (last_hidden * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            embs = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

        return embs.cpu().numpy()  # [batch, hidden]

    # ========= 主流程 =========

    def run(self):
        """
        主流程：加载数据 -> BERT 编码 -> 保存结果
        """
        # 1. 加载数据（CSV or THUCNews）
        df, texts = self.load_texts()
        total = len(texts)

        # 2. 加载模型
        self.build_model_and_tokenizer()

        max_length = self.config["max_length"]
        batch_size = self.config["batch_size"]
        self.logger.info(
            f"[Embedding] 开始编码，batch_size={batch_size}, max_length={max_length}"
        )

        # 3. 逐批编码
        all_embeddings = []
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]

            embs = self.encode_batch(batch_texts)
            all_embeddings.append(embs)

            self.logger.info(f"[Embedding] 已编码 {end}/{total}")

        all_embeddings = np.vstack(all_embeddings)
        self.logger.info(f"[Embedding] 向量形状: {all_embeddings.shape}")  # (N, hidden_size)

        # 4. 取 id / label
        id_column = self.config["id_column"]
        label_column = self.config["label_column"]
        ids = df[id_column].to_numpy()
        labels = df[label_column].to_numpy()

        # 5. 保存文件
        output_embeddings = self.config["output_embeddings"]
        output_npz = self.config["output_npz"]
        output_meta = self.config["output_meta"]

        np.save(output_embeddings, all_embeddings)
        np.savez(
            output_npz,
            ids=ids,             # [N]
            embeddings=all_embeddings,  # [N, hidden]
            labels=labels,       # [N]
        )
        self.logger.info(f"[Embedding] 已保存向量到: {output_embeddings}")
        self.logger.info(f"[Embedding] 已保存 npz 到: {output_npz}")

        # 保留完整 meta，后续聚类/分析用
        df.to_csv(output_meta, index=False)
        self.logger.info(f"[Embedding] 已保存元信息到: {output_meta}")

        return df, all_embeddings


if __name__ == "__main__":
    embedder = CsvBertEmbedder(CONFIG.get("EMBEDDING"))
    embedder.run()

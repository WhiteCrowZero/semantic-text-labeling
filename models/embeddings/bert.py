# from transformers import AutoTokenizer, AutoModel
# import torch
# import numpy as np
#
# # 你可以换成别的，例如 'bert-base-uncased'、'hfl/chinese-bert-wwm-ext' 等
# MODEL_NAME = "bert-base-chinese"
#
# # 加载 tokenizer 和 模型
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModel.from_pretrained(MODEL_NAME)
#
# def encode_sentences_with_cls(sentences, max_length=64, device=None, normalize=True):
#     """
#     用 BERT 输出句子级 CLS 语义向量。
#
#     参数:
#         sentences: list[str] 或 单个 str
#         max_length: 最大长度，超过会截断
#         device: 'cuda' / 'cpu'，默认自动选
#         normalize: 是否做 L2 归一化（做相似度/聚类时建议 True）
#
#     返回:
#         np.ndarray, shape = (batch_size, hidden_size)
#     """
#     if isinstance(sentences, str):
#         sentences = [sentences]
#
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     model.to(device)
#     model.eval()
#
#     # 编码文本 -> BERT 输入
#     encoded = tokenizer(
#         sentences,
#         padding=True,
#         truncation=True,
#         max_length=max_length,
#         return_tensors="pt"
#     )
#
#     encoded = {k: v.to(device) for k, v in encoded.items()}
#
#     with torch.no_grad():
#         outputs = model(**encoded)
#         # outputs.last_hidden_state: [batch_size, seq_len, hidden_size]
#         # 取第 0 个 token（[CLS]）作为句子向量
#         cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
#
#         if normalize:
#             cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
#
#     return cls_embeddings.cpu().numpy()
#
#
# if __name__ == "__main__":
#     sents = [
#         "我很喜欢这款手机，速度很快，外观也好看。",
#         "这个手机太卡了，电池还不耐用，再也不会买了。",
#         "我服了这是什么东西?垃圾粗粮"
#     ]
#
#     vecs = encode_sentences_with_cls(sents, max_length=64)
#     print("向量形状:", vecs.shape)   # (2, 768)
#     print("第一条句子的前 10 维:", vecs[0][:10])

# bert_encoder_service.py
"""
功能：
  - 从 CSV 加载数据集
  - 用 BERT 提取句子级 CLS 语义向量
  - 把向量保存成 .npy，元信息保存成 .csv（给组员用）

使用方式：
  1. 修改下面 CONFIG 里面的路径和列名
  2. 在 PyCharm 里直接运行这个文件
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple

# ========= 配置区：你只需要改这里 =========
CONFIG = {
    # 模型：推荐中文 RoBERTa WWM，比 bert-base-chinese 更强一些
    "model_name": "hfl/chinese-roberta-wwm-ext",

    # 输入数据集：CSV 路径
    "input_csv": r"train.csv",   # 改成你自己的路径

    # 文本所在的列名
    "text_column": "content",

    # 输出：语义向量文件
    "output_embeddings": r"output/bert_embeddings.npy",

    "id_column": "content_id",
    "label_column": "sentiment_value",   # 比如：-1/0/1 或 0/1/2


    # 新增：带 meta 的 npz 文件
    "output_npz": r"output/bert_with_id_label.npz",

    # 输出：元信息（和原始 df 一样，只是顺序固定）
    "output_meta": r"output/reviews_meta.csv",

    # BERT 相关参数
    "max_length": 64,
    "batch_size": 32,
}
# ========================================


def load_texts_from_csv(csv_path: str, text_column: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    从 CSV 加载文本和元信息（id、label 等）。
    要求 CSV 至少有一列文本列 text_column。
    """
    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise ValueError(
            f"CSV 中找不到文本列 '{text_column}'，现有列: {df.columns.tolist()}"
        )
    texts = df[text_column].astype(str).tolist()
    return df, texts


def build_model_and_tokenizer(model_name: str, device: str = None):
    """
    加载 tokenizer 和 BERT 模型。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer, device


def encode_batch(
    model,
    tokenizer,
    sentences: List[str],
    max_length: int,
    device: str,
) -> np.ndarray:
    """
    对一批句子做 BERT 编码，返回 CLS 向量（已 L2 归一化）。
    """
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]
        cls = last_hidden[:, 0, :]               # 取 CLS
        cls = torch.nn.functional.normalize(cls, p=2, dim=1)  # L2 归一化

    return cls.cpu().numpy()  # [batch, hidden]


def encode_csv_to_embeddings(config: dict):
    """
    主流程：加载 CSV -> BERT 编码 -> 保存结果
    """
    model_name = config["model_name"]
    input_csv = config["input_csv"]
    # text_column = config["text_column"]
    # output_embeddings = config["output_embeddings"]
    # output_meta = config["output_meta"]
    max_length = config["max_length"]
    batch_size = config["batch_size"]
    text_column = config["text_column"]
    id_column = config["id_column"]
    label_column = config["label_column"]
    output_embeddings = config["output_embeddings"]
    output_meta = config["output_meta"]
    output_npz = config["output_npz"]

    # 1. 加载数据
    print(f"加载数据集: {input_csv}")
    df, texts = load_texts_from_csv(input_csv, text_column)
    total = len(texts)
    print(f"样本数: {total}")

    # 2. 加载模型
    print(f"加载模型: {model_name}")
    model, tokenizer, device = build_model_and_tokenizer(model_name)
    print(f"使用设备: {device}")

    # 3. 逐批编码
    all_embeddings = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        embs = encode_batch(
            model=model,
            tokenizer=tokenizer,
            sentences=batch_texts,
            max_length=max_length,
            device=device,
        )
        all_embeddings.append(embs)
        print(f"\r已编码 {end}/{total}", end="")
    print()

    all_embeddings = np.vstack(all_embeddings)
    print("向量形状:", all_embeddings.shape)  # (N, hidden_size)

    ids = df[id_column].to_numpy()
    labels = df[label_column].to_numpy()

    # 4. 保存文件
    # 4.1 保存 .npy 语义向量
    np.save(output_embeddings, all_embeddings)
    np.savez(
        output_npz,
        ids=ids,  # [N]
        embeddings=all_embeddings,  # [N, hidden]
        labels=labels  # [N]
    )
    print(f"已保存向量到: {output_embeddings}")

    # 4.2 保存元信息（df 原样保存，保证顺序和向量一致）
    df.to_csv(output_meta, index=False)
    print(f"已保存元信息到: {output_meta}")

    return df, all_embeddings


if __name__ == "__main__":
    # 在 PyCharm 里直接运行这个文件即可
    encode_csv_to_embeddings(CONFIG)


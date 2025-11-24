# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class BertConfig:
#     def __init__(
#         self,
#         vocab_size=30522,
#         hidden_size=256,
#         num_hidden_layers=4,
#         num_attention_heads=4,
#         intermediate_size=512,
#         max_position_embeddings=512,
#         type_vocab_size=2,
#         dropout_prob=0.1,
#     ):
#         self.vocab_size = vocab_size
#         self.hidden_size = hidden_size
#         self.num_hidden_layers = num_hidden_layers
#         self.num_attention_heads = num_attention_heads
#         self.intermediate_size = intermediate_size
#         self.max_position_embeddings = max_position_embeddings
#         self.type_vocab_size = type_vocab_size
#         self.dropout_prob = dropout_prob
#
#
# class BertEmbeddings(nn.Module):
#     """
#     token + position + segment 的标准三合一 embedding
#     """
#
#     def __init__(self, config: BertConfig):
#         super().__init__()
#         self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
#         self.position_embeddings = nn.Embedding(
#             config.max_position_embeddings, config.hidden_size
#         )
#         self.token_type_embeddings = nn.Embedding(
#             config.type_vocab_size, config.hidden_size
#         )
#
#         self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
#         self.dropout = nn.Dropout(config.dropout_prob)
#
#     def forward(self, input_ids, token_type_ids=None):
#         # input_ids: (B, L)
#         batch_size, seq_len = input_ids.size()
#
#         if token_type_ids is None:
#             token_type_ids = torch.zeros_like(input_ids)
#
#         # 位置编号: [0, 1, 2, ... L-1]
#         position_ids = torch.arange(
#             seq_len, dtype=torch.long, device=input_ids.device
#         ).unsqueeze(0).expand(batch_size, seq_len)
#
#         word_embeddings = self.word_embeddings(input_ids)
#         position_embeddings = self.position_embeddings(position_ids)
#         token_type_embeddings = self.token_type_embeddings(token_type_ids)
#
#         embeddings = word_embeddings + position_embeddings + token_type_embeddings
#         embeddings = self.layer_norm(embeddings)
#         embeddings = self.dropout(embeddings)
#         return embeddings
#
#
# class BertSelfAttention(nn.Module):
#     def __init__(self, config: BertConfig):
#         super().__init__()
#         if config.hidden_size % config.num_attention_heads != 0:
#             raise ValueError("hidden_size 必须能整除 num_attention_heads")
#
#         self.num_heads = config.num_attention_heads
#         self.head_dim = config.hidden_size // config.num_attention_heads
#         self.all_head_size = config.hidden_size
#
#         self.q = nn.Linear(config.hidden_size, self.all_head_size)
#         self.k = nn.Linear(config.hidden_size, self.all_head_size)
#         self.v = nn.Linear(config.hidden_size, self.all_head_size)
#
#         self.dropout = nn.Dropout(config.dropout_prob)
#
#     def _transpose_for_scores(self, x):
#         # x: (B, L, H) -> (B, num_heads, L, head_dim)
#         new_shape = (x.size(0), x.size(1), self.num_heads, self.head_dim)
#         x = x.view(*new_shape)
#         return x.permute(0, 2, 1, 3)
#
#     def forward(self, hidden_states, attention_mask=None):
#         # hidden_states: (B, L, H)
#         q = self._transpose_for_scores(self.q(hidden_states))
#         k = self._transpose_for_scores(self.k(hidden_states))
#         v = self._transpose_for_scores(self.v(hidden_states))
#
#         # 注意力分数: (B, num_heads, L, L)
#         attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
#
#         if attention_mask is not None:
#             # attention_mask 的形状: (B, 1, 1, L)，里面是 0 或 -1e4 这种大量负数
#             attn_scores = attn_scores + attention_mask
#
#         attn_probs = F.softmax(attn_scores, dim=-1)
#         attn_probs = self.dropout(attn_probs)
#
#         # 上下文: (B, num_heads, L, head_dim)
#         context = torch.matmul(attn_probs, v)
#         # -> (B, L, num_heads * head_dim) = (B, L, H)
#         context = context.permute(0, 2, 1, 3).contiguous()
#         new_shape = (context.size(0), context.size(1), self.num_heads * self.head_dim)
#         context = context.view(*new_shape)
#         return context
#
#
# class BertSelfOutput(nn.Module):
#     def __init__(self, config: BertConfig):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
#         self.dropout = nn.Dropout(config.dropout_prob)
#
#     def forward(self, hidden_states, input_tensor):
#         # 残差 + LN
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.layer_norm(hidden_states + input_tensor)
#         return hidden_states
#
#
# class BertAttention(nn.Module):
#     def __init__(self, config: BertConfig):
#         super().__init__()
#         self.self = BertSelfAttention(config)
#         self.output = BertSelfOutput(config)
#
#     def forward(self, hidden_states, attention_mask=None):
#         self_output = self.self(hidden_states, attention_mask)
#         attention_output = self.output(self_output, hidden_states)
#         return attention_output
#
#
# class BertIntermediate(nn.Module):
#     def __init__(self, config: BertConfig):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
#         self.act = nn.GELU()  # BERT 原版是 GELU
#
#     def forward(self, hidden_states):
#         return self.act(self.dense(hidden_states))
#
#
# class BertOutput(nn.Module):
#     def __init__(self, config: BertConfig):
#         super().__init__()
#         self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
#         self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
#         self.dropout = nn.Dropout(config.dropout_prob)
#
#     def forward(self, hidden_states, input_tensor):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.layer_norm(hidden_states + input_tensor)
#         return hidden_states
#
#
# class BertLayer(nn.Module):
#     def __init__(self, config: BertConfig):
#         super().__init__()
#         self.attention = BertAttention(config)
#         self.intermediate = BertIntermediate(config)
#         self.output = BertOutput(config)
#
#     def forward(self, hidden_states, attention_mask=None):
#         attention_output = self.attention(hidden_states, attention_mask)
#         intermediate_output = self.intermediate(attention_output)
#         layer_output = self.output(intermediate_output, attention_output)
#         return layer_output
#
#
# class BertEncoder(nn.Module):
#     def __init__(self, config: BertConfig):
#         super().__init__()
#         self.layers = nn.ModuleList(
#             [BertLayer(config) for _ in range(config.num_hidden_layers)]
#         )
#
#     def forward(self, hidden_states, attention_mask=None):
#         for layer in self.layers:
#             hidden_states = layer(hidden_states, attention_mask)
#         return hidden_states  # (B, L, H)
#
#
# class BertForSentenceEmbedding(nn.Module):
#     """
#     只做 encoder + CLS 池化的句向量 BERT
#     forward 默认返回: (B, H) 的句子级语义向量
#     """
#
#     def __init__(self, config: BertConfig):
#         super().__init__()
#         self.config = config
#         self.embeddings = BertEmbeddings(config)
#         self.encoder = BertEncoder(config)
#
#         # pooler: 可以理解为对 CLS 做一个仿 BERT 的 Tanh 投影
#         self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
#         self.pooler_activation = nn.Tanh()
#
#     def forward(
#         self,
#         input_ids,
#         attention_mask=None,
#         token_type_ids=None,
#         return_sequence=False,
#     ):
#         """
#         input_ids: (B, L), int64
#         attention_mask: (B, L), 1 表示有内容，0 表示 padding
#         token_type_ids: (B, L)，句子对任务时区分 segment A/B，否则全 0 即可
#
#         return:
#             默认: pooled_cls: (B, H)
#             若 return_sequence=True: (pooled_cls, sequence_output)
#         """
#         # 如果没给 mask，就默认 pad_id == 0
#         if attention_mask is None:
#             attention_mask = (input_ids != 0).long()
#
#         # (B,L) -> (B,1,1,L)，0 的位置加一个巨大负数，softmax 后接近 0
#         extended_mask = attention_mask[:, None, None, :].to(
#             dtype=input_ids.dtype, device=input_ids.device
#         )
#         extended_mask = (1.0 - extended_mask) * -10000.0
#
#         embedding_output = self.embeddings(input_ids, token_type_ids)
#         sequence_output = self.encoder(embedding_output, extended_mask)
#
#         # 取 CLS: 第 0 个位置
#         cls_token = sequence_output[:, 0]  # (B, H)
#
#         # 仿 BERT pooler（你也可以直接用 cls_token 不过一层线性+tanh 通常更稳定一点）
#         pooled_output = self.pooler_activation(self.pooler(cls_token))
#
#         if return_sequence:
#             return pooled_output, sequence_output
#         return pooled_output
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import jieba
from collections import Counter
from typing import List, Dict, Tuple

# =======================
# 1. 分词 & 词表构建
# =======================

def tokenize(text: str) -> List[str]:
    """使用 jieba 对中文句子做分词。"""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return list(jieba.lcut(text.strip()))

def build_vocab(texts: List[str],
                min_freq: int = 1,
                max_vocab_size: int = 50000) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    基于数据集构建词表，只收集出现频率 >= min_freq 的词，最多 max_vocab_size 个。
    保留三个特殊符号:
        [PAD] -> 0
        [UNK] -> 1
    （这里不强制搞词级 CLS token，CLS 句向量我们用 pooling 得到）
    """
    counter = Counter()
    for t in texts:
        tokens = tokenize(t)
        counter.update(tokens)

    # 预留特殊 token
    word2id = {"[PAD]": 0, "[UNK]": 1}
    for w, c in counter.most_common():
        if c < min_freq:
            continue
        if w in word2id:
            continue
        if len(word2id) >= max_vocab_size:
            break
        word2id[w] = len(word2id)

    id2word = {i: w for w, i in word2id.items()}
    print(f"词表大小（含特殊词）：{len(word2id)}")
    return word2id, id2word

def texts_to_ids(texts: List[str],
                 word2id: Dict[str, int],
                 max_length: int) -> torch.Tensor:
    """
    把一批句子 -> id 矩阵 [batch, max_length]
    用 [PAD] 补齐，不加 CLS token，后面用 pooling 做句向量。
    """
    pad_id = word2id["[PAD]"]
    unk_id = word2id["[UNK]"]

    all_ids = []
    for t in texts:
        tokens = tokenize(t)
        tokens = tokens[:max_length]
        ids = [word2id.get(tok, unk_id) for tok in tokens]
        if len(ids) < max_length:
            ids = ids + [pad_id] * (max_length - len(ids))
        all_ids.append(ids)

    return torch.tensor(all_ids, dtype=torch.long)

# =======================
# 2. 加载预训练中文词向量
# =======================

def load_pretrained_embeddings(
    vec_path: str,
    word2id: Dict[str, int],
    embedding_dim: int = 300,
) -> torch.FloatTensor:
    """
    只从预训练词向量文件中加载我们词表里用得到的词。
    文件格式一般是：word val1 val2 ... val300 （第一行可能是 vocab_size dim，要跳过）
    """
    vocab_size = len(word2id)
    emb_matrix = np.random.normal(scale=0.01, size=(vocab_size, embedding_dim)).astype("float32")

    # [PAD] 用 0 向量，[UNK] 先随机，后面可以再处理
    pad_id = word2id["[PAD]"]
    emb_matrix[pad_id] = np.zeros(embedding_dim, dtype="float32")

    # 建一个快速查找表
    needed = set(word2id.keys())
    found = 0

    print(f"开始从预训练词向量中筛选，文件: {vec_path}")
    with open(vec_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # 有些文件第一行是 "vocab_size dim"
            if len(parts) == 2:
                # 尝试解析为数字，如果成功就跳过这一行
                try:
                    _ = int(parts[0])
                    _ = int(parts[1])
                    continue
                except ValueError:
                    pass
            word = parts[0]
            if word not in needed:
                continue
            vec = parts[1:]
            if len(vec) != embedding_dim:
                continue
            emb_matrix[word2id[word]] = np.asarray(vec, dtype="float32")
            found += 1

    print(f"在预训练向量中找到了 {found} / {len(needed)} 个词的向量。")

    # 如果想让 [UNK] 更合理一点，可以用已加载词向量的均值
    unk_id = word2id["[UNK]"]
    if found > 0:
        mean_vec = emb_matrix[2:2+found].mean(axis=0)  # 略粗糙，但够用
        emb_matrix[unk_id] = mean_vec

    return torch.from_numpy(emb_matrix)

# =======================
# 3. 多头自注意力 + 句向量
# =======================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B, L, d_model]
        attn_mask: [B, 1, 1, L]，pad 位置为 -1e9，其余为 0
        """
        B, L, _ = x.size()

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        def split_heads(t):
            # [B, L, d_model] -> [B, h, L, d_head]
            return t.view(B, L, self.num_heads, self.d_head).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)  # [B,h,L,L]

        if attn_mask is not None:
            scores = scores + attn_mask

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)  # [B,h,L,d_head]
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.W_o(context)
        return out

class AttnSentenceEncoder(nn.Module):
    """
    结构：
      - 预训练词向量 embedding（300d）
      - 一层 Multi-Head Self-Attention + 残差 + LayerNorm（d_model = 300）
      - mean pooling 得到 300 维句向量
      - 0 填充扩展成 768 维（前 300 维有信息，后面补 0）
    """
    def __init__(self,
                 embedding_matrix: torch.FloatTensor,
                 pad_id: int = 0,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        vocab_size, emb_dim = embedding_matrix.size()
        self.pad_id = pad_id
        self.d_model = emb_dim  # 这里就是 300

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.embedding.weight.data.copy_(embedding_matrix)
        # 语义都来自预训练词向量，如果不想微调，就冻结
        self.embedding.weight.requires_grad = False

        self.attn = MultiHeadSelfAttention(self.d_model, num_heads, dropout)
        self.ln = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, L]
        返回:
          sent_vec_768: [B, 768]  句子级语义向量
        """
        B, L = input_ids.size()
        device = input_ids.device

        x = self.embedding(input_ids)  # [B,L,d_model]

        # 构造 padding mask
        pad_mask = (input_ids == self.pad_id).unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
        attn_mask = pad_mask.to(x.dtype) * -1e9

        # 一层自注意力 + 残差 + LN
        attn_out = self.attn(x, attn_mask)
        x = self.ln(x + self.dropout(attn_out))  # [B,L,d_model]

        # 对非 PAD 位置做 mean pooling 得到句向量（相当于“CLS”）
        pad_mask_float = (input_ids != self.pad_id).float()  # [B,L]
        # 避免除 0
        lengths = pad_mask_float.sum(dim=1, keepdim=True).clamp(min=1.0)
        sent_vec = (x * pad_mask_float.unsqueeze(-1)).sum(dim=1) / lengths  # [B,d_model]

        # 扩展到 768 维：前 300 维为语义向量，后面补 0（不是随机投影）
        if self.d_model < 768:
            pad_dim = 768 - self.d_model
            zeros = torch.zeros(B, pad_dim, device=device, dtype=sent_vec.dtype)
            sent_vec_768 = torch.cat([sent_vec, zeros], dim=-1)
        else:
            sent_vec_768 = sent_vec[:, :768]

        return sent_vec_768

# =======================
# 4. 对 CSV 编码：句子 -> CLS 向量
# =======================

def encode_sentences_with_cls(
    model: AttnSentenceEncoder,
    word2id: Dict[str, int],
    sentences: List[str],
    max_length: int = 64,
    device: str = "cpu",
    normalize: bool = True,
    batch_size: int = 64,
) -> np.ndarray:
    """
    类似你原来 bert.py 里的接口：
      输入一批句子 -> 输出 [N, 768] 的句子向量（CLS）
    """
    model.eval()
    all_vecs = []

    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            batch_sents = sentences[start:start + batch_size]
            ids = texts_to_ids(batch_sents, word2id, max_length=max_length)  # [B,L]
            ids = ids.to(device)
            cls_vec = model(ids)  # [B,768]
            if normalize:
                cls_vec = torch.nn.functional.normalize(cls_vec, p=2, dim=1)
            all_vecs.append(cls_vec.cpu())

    all_vecs = torch.cat(all_vecs, dim=0).numpy()
    return all_vecs

def load_texts_from_csv(csv_path: str, text_column: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(csv_path)
    texts = df[text_column].astype(str).tolist()
    return df, texts

def encode_csv_to_embeddings(config: Dict):
    """
    模仿你原来的 encode_csv_to_embeddings，只是把底层换成我们手搓的 AttnSentenceEncoder。
    """
    input_csv = config["input_csv"]
    text_column = config["text_column"]
    output_embeddings = config["output_embeddings"]
    output_meta = config["output_meta"]
    max_length = config.get("max_length", 64)
    batch_size = config.get("batch_size", 64)

    pretrained_vec_path = config["pretrained_vec_path"]
    embedding_dim = config.get("embedding_dim", 300)
    min_freq = config.get("min_freq", 1)
    max_vocab_size = config.get("max_vocab_size", 50000)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 1. 加载数据
    print(f"加载数据集: {input_csv}")
    df, texts = load_texts_from_csv(input_csv, text_column)
    total = len(texts)
    print(f"样本数: {total}")

    # 2. 基于数据集构建词表
    print("构建词表...")
    word2id, id2word = build_vocab(texts, min_freq=min_freq, max_vocab_size=max_vocab_size)

    # 3. 加载预训练中文词向量（只加载词表中用到的）
    emb_matrix = load_pretrained_embeddings(pretrained_vec_path, word2id, embedding_dim=embedding_dim)

    # 4. 构建句向量编码器
    model = AttnSentenceEncoder(
        embedding_matrix=emb_matrix,
        pad_id=word2id["[PAD]"],
        num_heads=config.get("num_heads", 4),
        dropout=config.get("dropout", 0.1),
    ).to(device)

    # 5. 编码所有句子
    print("开始编码句子为 CLS 语义向量...")
    all_embeddings = encode_sentences_with_cls(
        model=model,
        word2id=word2id,
        sentences=texts,
        max_length=max_length,
        device=device,
        normalize=True,
        batch_size=batch_size,
    )
    print("编码完成，向量形状:", all_embeddings.shape)

    # 6. 保存结果
    os.makedirs(os.path.dirname(output_embeddings), exist_ok=True)
    np.save(output_embeddings, all_embeddings)
    print(f"已保存向量到: {output_embeddings}")

    df.to_csv(output_meta, index=False)
    print(f"已保存元信息到: {output_meta}")

    return df, all_embeddings

# =======================
# 5. 配置
# =======================

CONFIG = {
    # 你下载的预训练中文词向量路径（Mixed-large 综合 Word 300d 解压后得到的文件）
    # 举例： "data/sgns.merge.word"
    "pretrained_vec_path": r"sgns.merge.word/sgns.merge.word",

    # 输入 CSV & 文本列名
    "input_csv": r"train.csv",
    "text_column": "content",

    # 输出
    "output_embeddings": r"output_1/attn_bert_embeddings.npy",
    "output_meta": r"output_1/reviews_meta_attn.csv",

    # 模型 & 训练相关参数
    "embedding_dim": 300,      # 预训练词向量维度
    "max_length": 64,          # 句子最大长度（超过截断，短了 padding）
    "batch_size": 64,
    "min_freq": 1,             # 词频阈值
    "max_vocab_size": 50000,   # 词表最大容量
    "num_heads": 4,
    "dropout": 0.1,
}

if __name__ == "__main__":
    encode_csv_to_embeddings(CONFIG)

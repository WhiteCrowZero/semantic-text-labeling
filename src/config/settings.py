import os
import dotenv

dotenv.load_dotenv()

# ========= 项目基本路径 =========
BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)
# print(BASE_DIR)
# print(PROJECT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")

# ========= 配置区 =========
CONFIG = {
    "EMBEDDING": {},
    "SENTIMENT": {},
    "CLUSTERING": {},
    "NLP": {},
}

# ========= embedding =========
EMBEDDING = {
    # 模型：推荐中文 RoBERTa WWM，比 bert-base-chinese 更强一些
    "model_name": "hfl/chinese-roberta-wwm-ext",
    # 若指定本地模型路径，则优先从本地加载；为空则使用 model_name 从 HuggingFace 下载
    "local_model_path": os.path.join(MODEL_DIR, "embeddings", "chinese-roberta-wwm-ext"),
    # 设备：cuda / cpu，留空则自动检测
    "device": os.getenv("DEVICE", ""),
    # BERT 相关参数
    "max_length": 64,
    "batch_size": 32,
}

# ========= sentiment =========
SENTIMENT = {}

# ========= clustering =========
CLUSTERING = {
    "PCA_COMPONENTS": 30,  # PCA 降维维度
    "KMEANS_K": 20,  # KMeans 聚类数上限
    "FIG_DIR": os.path.join(DATA_DIR, "output", "figures"),
    "DEFAULT_ALGO": "both",   # 默认算法类型 kmeans 或 hdbscan 或 both
}

NLP = {
    # NER
    "processors": "tokenize,pos,lemma,ner,depparse",
    "lang": "zh",
    "use_gpu": False,
    "local_model_path": os.path.join(MODEL_DIR, "ner"),
    # LLM
    "model_name": "gpt-4o",
    "max_tokens": 2000,
    "temperature": 0.7,
}

# ========= 重载配置 =========
CONFIG["EMBEDDING"] = EMBEDDING
CONFIG["SENTIMENT"] = SENTIMENT
CONFIG["CLUSTERING"] = CLUSTERING
CONFIG["NLP"] = NLP

import os
import dotenv

dotenv.load_dotenv()

# ========= 项目基本路径 =========
BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")

# ========= 配置区 =========
CONFIG = {
    "EMBEDDING": {},
    "SENTIMENT": {},
    "CLUSTERING": {},
}

# ========= embedding =========
EMBEDDING = {
    # 模型：推荐中文 RoBERTa WWM，比 bert-base-chinese 更强一些
    "model_name": os.getenv("MODEL_NAME", "hfl/chinese-roberta-wwm-ext"),
    # 若指定本地模型路径，则优先从本地加载；为空则使用 model_name 从 HuggingFace 下载
    "local_model_path": os.getenv(
        "LOCAL_MODEL_PATH", ""
    ),  # 例如: /path/to/local/model/dir
    # 设备：cuda / cpu，留空则自动检测
    "device": os.getenv("DEVICE", ""),
    # BERT 相关参数
    "max_length": 64,
    "batch_size": 32,
    # 输入数据集：CSV 路径
    "input_csv": os.getenv(
        "INPUT_CSV",
        os.path.join(DATA_DIR, "input", "train.csv"),
    ),
    # 数据集信息
    "text_column": "content",
    "id_column": "content_id",
    "label_column": "sentiment_value",  # 比如：-1/0/1 或 0/1/2
    # 语义向量文件
    "output_embeddings": os.getenv(
        "OUTPUT_EMBEDDING",
        os.path.join(DATA_DIR, "output", "bert_embeddings.npy"),
    ),
    # 带 meta 的 npz 文件
    "output_npz": os.getenv(
        "OUTPUT_NPZ",
        os.path.join(DATA_DIR, "output", "bert_with_id_label.npz"),
    ),
    # 元信息（和原始 df 一样，只是顺序固定）
    "output_meta": os.getenv(
        "OUTPUT_META",
        os.path.join(DATA_DIR, "output", "reviews_meta.csv"),
    ),
}

# ========= sentiment =========
SENTIMENT = {}

# ========= clustering =========
CLUSTERING = {}

# ========= 重载配置 =========
CONFIG["EMBEDDING"] = EMBEDDING
CONFIG["SENTIMENT"] = SENTIMENT
CONFIG["CLUSTERING"] = CLUSTERING

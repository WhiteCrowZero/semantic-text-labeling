# src/services/sentiment_service.py
from typing import List
from src.utils.logger import logger


class SentimentService:
    """
    后续可以接：
      - 传统 ML 分类器（LogReg / XGBoost）
      - 或者直接调用 LLM 接口做情感分类
    """
    def __init__(self):
        # TODO: 加载模型 / 初始化 client
        ...

    def predict_batch(self, texts: List[str]):
        # TODO: 真正实现
        logger.info(f"[SentimentService] 收到 {len(texts)} 条文本进行情感分析")
        return [
            {"sentiment": "neutral", "score": 0.5} for _ in texts
        ]

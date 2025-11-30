# src/api/v1/sentiment.py
from fastapi import APIRouter, Depends

from src.schemas.sentiment import SentimentRequest, SentimentResponse, SentimentResult
from src.services.sentiment_service import SentimentService
from src.services.embeddings_service import EmbeddingService, get_embedding_service

router = APIRouter()

# 简单依赖（你也可以像 embedding 一样搞个 get_sentiment_service）
sentiment_service = SentimentService()


@router.post("/analyze", response_model=SentimentResponse)
def analyze_sentiment(
    payload: SentimentRequest,
    embed_svc: EmbeddingService = Depends(get_embedding_service),
):
    texts = [item.text for item in payload.items]

    # 1）如果没有传 embedding，则在服务端统一编码
    if payload.items and payload.items[0].embedding is None:
        _ = embed_svc.encode(texts)  # 如果你需要 embedding 做特征，这里可以传给模型

    # 2）调用情感模型
    raw_results = sentiment_service.predict_batch(texts)

    results = [
        SentimentResult(sentiment=r["sentiment"], score=r["score"])
        for r in raw_results
    ]
    return SentimentResponse(results=results)

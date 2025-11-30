
# src/schemas/sentiment.py
from typing import List, Optional
from pydantic import BaseModel


class SentimentItem(BaseModel):
    text: str
    embedding: Optional[list] = None  # 可选：外部已算好向量


class SentimentRequest(BaseModel):
    items: List[SentimentItem]


class SentimentResult(BaseModel):
    sentiment: str        # "positive" / "negative" / "neutral"
    score: float          # 置信度


class SentimentResponse(BaseModel):
    results: List[SentimentResult]

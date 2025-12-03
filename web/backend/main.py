# src/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web.backend.api.v1 import embeddings, sentiment, clustering, tagging, review

app = FastAPI(
    title="Semantic Workflow Platform",
    version="0.1.0",
)

# CORS 配置（前后端分离用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由分模块挂载
app.include_router(embeddings.router, prefix="/api/v1/embeddings", tags=["Embeddings"])
app.include_router(sentiment.router, prefix="/api/v1/sentiment", tags=["Sentiment"])
app.include_router(clustering.router, prefix="/api/v1/clustering", tags=["Clustering"])
app.include_router(tagging.router, prefix="/api/v1/tagging", tags=["Tagging"])
app.include_router(review.router, prefix="/api/v1/review", tags=["Review"])


@app.get("/health")
def health_check():
    return {"status": "ok"}

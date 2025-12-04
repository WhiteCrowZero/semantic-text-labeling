from pydantic import BaseModel
from typing import List, Optional, Dict, Any


# ------------------------------------------------------
# 抽象类
# ------------------------------------------------------
class BaseResponse(BaseModel):
    success: bool = True
    message: Optional[str] = None


# ------------------------------------------------------
# ner
# ------------------------------------------------------
class NERRequest(BaseModel):
    text: str


class Entity(BaseModel):
    text: str
    type: str


class AdjPair(BaseModel):
    subject: str
    adjective: str


class NERResponse(BaseModel):
    text: str
    entities: List[Entity]
    subjects: List[str]
    adj_pairs: List[AdjPair]
    main_event: Optional[str]
    keywords: List[str]


# ------------------------------------------------------
# sentiment
# ------------------------------------------------------


# ------------------------------------------------------
# tag(llm+clustering)
# ------------------------------------------------------
class TagSubmitRequest(BaseModel):
    file_hash: str
    algo: Optional[str] = None
    kmeans_k: Optional[int] = None


class TagSubmitResponse(BaseModel):
    task_id: str
    message: str = "任务已提交，请稍后查询状态"


class TagStatusResponse(BaseModel):
    task_id: str
    status: str  # pending / running / finished / failed
    error: Optional[str] = None


class ClusterResult(BaseModel):
    """单个聚类算法的结果"""

    labels: List[int]  # 聚类标签
    metrics: Dict[str, Any]  # 评估指标
    fig_path: Optional[str] = None  # 可视化图片路径


class TagResultResponse(BaseModel):
    """标签分析任务的完整返回结果"""

    # 每个文本的原始标签/关键词
    text_tags: List[str]
    # 聚类分析结果（多个算法）
    clustering_results: Dict[str, ClusterResult]
    # 额外的元数据
    metadata: Optional[Dict[str, Any]] = None


# ------------------------------------------------------
# tools
# ------------------------------------------------------
class UploadResponse(BaseModel):
    success: bool
    file_hash: Optional[str] = None
    count: Optional[int] = None
    message: Optional[str] = None

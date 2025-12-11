# -*- coding: utf-8 -*-
# src/main.py

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

import codecs

# 修改默认错误处理为忽略
codecs.register_error("strict", codecs.ignore_errors)

import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from src.common.file_utils import parse_text_file, generate_hash
from web.backend.api.v1.services import TextModelService
from web.backend.api.v1.schemas import (
    NERRequest,
    NERResponse,
    Entity,
    AdjPair,
    UploadResponse,
    TagSubmitResponse,
    TagSubmitRequest,
    TagStatusResponse,
    TagResultResponse,
    ClusterResult,
)

# 缓存系统和消息队列系统
# 简化设计，未引入 Redis、MQ，直接使用内存
FILE_CACHE: Dict[str, list[str]] = {}
TASK_STATUS: Dict[str, dict] = {}
TASK_RESULT: Dict[str, dict] = {}

# 线程池
executor = ThreadPoolExecutor(max_workers=4)


app = FastAPI(
    title="Semantic Workflow Platform",
    version="0.1.0",
)

# CORS（前后端分离）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化服务（只加载一次）
model_service = TextModelService()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ner", response_model=NERResponse)
def ner_api(req: NERRequest):
    try:
        raw = model_service.ner_analysis(req.text)

        entities = [Entity(text=e[0], type=e[1]) for e in raw["entities"]]
        adj_pairs = [AdjPair(subject=p[0], adjective=p[1]) for p in raw["adj_pairs"]]

        return NERResponse(
            text=raw["text"],
            entities=entities,
            subjects=raw["subjects"],
            adj_pairs=adj_pairs,
            main_event=raw["main_event"],
            keywords=raw["keywords"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_tag_job(task_id: str, req: TagSubmitRequest):
    print(f"[DEBUG] Task {task_id} START")
    try:
        TASK_STATUS[task_id]["status"] = "running"

        texts = FILE_CACHE[req.file_hash]
        res = model_service.batch_tag_analysis(texts, req.algo, req.kmeans_k)
        tag_res = res.get("tags", [])
        clustering_res = res.get("clustering_res", {})

        # 构造返回结果
        result = TagResultResponse(
            text_tags=tag_res,  # 每个文本的标签结果
            clustering_results={
                "kmeans": (
                    ClusterResult(
                        labels=(
                            clustering_res["kmeans"]["labels"].tolist()
                            if isinstance(
                                clustering_res["kmeans"]["labels"], np.ndarray
                            )
                            else clustering_res["kmeans"]["labels"]
                        ),
                        metrics=clustering_res["kmeans"]["metrics"],
                        fig_path=clustering_res["kmeans"]["fig_path"],
                    )
                    if req.algo in ("kmeans", "both")
                    else None
                ),
                "hdbscan": (
                    ClusterResult(
                        labels=(
                            clustering_res["hdbscan"]["labels"].tolist()
                            if isinstance(
                                clustering_res["hdbscan"]["labels"], np.ndarray
                            )
                            else clustering_res["hdbscan"]["labels"]
                        ),
                        metrics=clustering_res["hdbscan"]["metrics"],
                        fig_path=clustering_res["hdbscan"]["fig_path"],
                    )
                    if req.algo in ("hdbscan", "both")
                    else None
                ),
            },
            metadata={
                "task_id": task_id,
                "n_samples": len(texts),
                "status": "finished",
            },
        )

        # 保存结果到全局缓存
        TASK_RESULT[task_id] = result.model_dump()
        TASK_STATUS[task_id]["status"] = "finished"
        print(f"[DEBUG] Task {task_id} DONE")
    except Exception as e:
        print(f"[DEBUG] Task {task_id} FAILED: {e}")
        TASK_STATUS[task_id]["status"] = "failed"
        TASK_STATUS[task_id]["error"] = str(e)

        # 返回一个错误状态的响应
        return TagResultResponse(
            text_tags=[],
            clustering_results={},
            metadata={"task_id": task_id, "status": "failed", "error": str(e)},
        )


@app.post("/tag/submit", response_model=TagSubmitResponse)
def submit_tag(req: TagSubmitRequest):
    if req.file_hash not in FILE_CACHE:
        raise HTTPException(status_code=400, detail="文件哈希不存在，请先上传文件")

    task_id = str(uuid.uuid4())
    TASK_STATUS[task_id] = {"status": "pending", "error": None}

    # 使用线程池提交
    executor.submit(run_tag_job, task_id, req)
    return TagSubmitResponse(task_id=task_id)


@app.get("/tag/status/{task_id}", response_model=TagStatusResponse)
def tag_status(task_id: str):
    if task_id not in TASK_STATUS:
        raise HTTPException(status_code=404, detail="任务不存在")

    entry = TASK_STATUS[task_id]
    return TagStatusResponse(
        task_id=task_id, status=entry["status"], error=entry["error"]
    )


@app.get("/tag/result/{task_id}", response_model=TagResultResponse)
def tag_result(task_id: str):
    if task_id not in TASK_RESULT:
        raise HTTPException(status_code=404, detail="结果不存在或任务未完成")

    res = TASK_RESULT[task_id]
    return TagResultResponse(**res)


@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...), column_name: str = Form("column_name")
):
    try:
        texts = parse_text_file(file, column_name)
        file_hash = generate_hash(texts)
        FILE_CACHE[file_hash] = texts

        return UploadResponse(
            success=True, file_hash=file_hash, count=len(texts), message="文件上传成功"
        )

    except Exception as e:
        return UploadResponse(success=False, message=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
    # uvicorn.run(app, host="0.0.0.0", port=8000)

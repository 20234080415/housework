from pathlib import Path
from uuid import uuid4

import redis
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tasks import generate_image


BASE_DIR = Path(__file__).resolve().parents[1]


def load_config():
    """读取项目配置。"""
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


config = load_config()
app = FastAPI(title="ControlNet Sketch-to-Image")
redis_client = redis.Redis(
    host=config["redis"]["host"],
    port=config["redis"]["port"],
    db=config["redis"]["db"],
    decode_responses=True,
)

# 开发阶段允许前端跨域访问。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    """图像生成请求参数。"""

    sketch_base64: str
    prompt: str
    steps: int = 20
    cfg_scale: float = 7.5
    cn_scale: float = 1.0


def ok(data=None, msg="ok"):
    """统一 API 成功返回格式。"""
    return {"code": 0, "data": data or {}, "msg": msg}


def api_response(code, data=None, msg="ok"):
    """统一 API 返回格式。"""
    return {"code": code, "data": data or {}, "msg": msg}


@app.post("/api/generate")
def create_generate_task(request: GenerateRequest):
    """创建异步生成任务。"""
    task_id = str(uuid4())
    redis_client.set(
        f"task:{task_id}:status",
        "pending",
        ex=config["redis"]["ttl"],
    )

    generate_image.delay(
        task_id,
        request.sketch_base64,
        request.prompt,
        request.steps,
        request.cfg_scale,
        request.cn_scale,
    )
    return ok({"task_id": task_id})


@app.get("/api/status/{task_id}")
def get_task_status(task_id: str):
    """查询异步生成任务状态。"""
    status = redis_client.get(f"task:{task_id}:status")

    if status is None:
        return api_response(404, {}, "task not found")

    data = {"status": status}
    if status == "done":
        data["result_base64"] = redis_client.get(f"task:{task_id}:result")

    return ok(data)


@app.get("/health")
def health():
    """健康检查。"""
    return {"status": "ok"}


if __name__ == "__main__":
    # 从配置文件读取服务监听地址和端口。
    uvicorn.run(
        "main:app",
        host=config["server"]["host"],
        port=config["server"]["port"],
        reload=True,
    )

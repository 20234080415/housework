from pathlib import Path

import yaml
from fastapi import FastAPI


BASE_DIR = Path(__file__).resolve().parents[1]


def load_config():
    """读取项目配置。"""
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


config = load_config()
app = FastAPI(title="ControlNet Sketch-to-Image")


def ok(data=None, msg="ok"):
    """统一 API 返回格式。"""
    return {"code": 0, "data": data or {}, "msg": msg}


@app.get("/health")
def health():
    return ok({"status": "running"})

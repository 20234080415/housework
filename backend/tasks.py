from pathlib import Path

import yaml
from celery import Celery


BASE_DIR = Path(__file__).resolve().parents[1]


def load_config():
    """读取 Celery 配置。"""
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


config = load_config()
celery_app = Celery(
    "controlnet_sketch",
    broker=config["celery"]["broker"],
    backend=config["celery"]["backend"],
)


@celery_app.task
def generate_image_task(payload):
    """异步生成任务占位。"""
    return {"status": "pending", "payload": payload}

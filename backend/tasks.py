import base64
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import redis
import yaml
from celery import Celery
from PIL import Image

from model import get_pipeline


BASE_DIR = Path(__file__).resolve().parents[1]


def load_config():
    """读取项目配置。"""
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


config = load_config()
celery_app = Celery(
    "controlnet_sketch",
    broker=config["celery"]["broker"],
    backend=config["celery"]["backend"],
)
redis_client = redis.Redis(
    host=config["redis"]["host"],
    port=config["redis"]["port"],
    db=config["redis"]["db"],
    decode_responses=True,
)


def _set_status(task_id, status):
    """写入任务状态。"""
    redis_client.set(f"task:{task_id}:status", status, ex=config["redis"]["ttl"])


def _decode_base64_image(sketch_base64):
    """将 base64 草图解码为 PIL 图像。"""
    if "," in sketch_base64:
        sketch_base64 = sketch_base64.split(",", 1)[1]

    image_bytes = base64.b64decode(sketch_base64)
    image_size = config["inference"]["image_size"]
    return Image.open(BytesIO(image_bytes)).convert("RGB").resize((image_size, image_size))


def _image_to_base64(image):
    """将 PIL 图像编码为 base64。"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _build_canny_image(sketch_image):
    """根据草图生成 Canny 边缘条件图。"""
    sketch_array = np.array(sketch_image)
    gray_image = cv2.cvtColor(sketch_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(
        gray_image,
        config["canny"]["low_threshold"],
        config["canny"]["high_threshold"],
    )
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


def _build_prompt(prompt):
    """组合用户提示词和配置中的质量提示词。"""
    prompt_suffix = config["inference"]["prompt_suffix"].strip()
    if not prompt_suffix:
        return prompt.strip()
    return f"{prompt.strip()}, {prompt_suffix}"


@celery_app.task(name="generate_image")
def generate_image(task_id, sketch_base64, prompt, steps, cfg_scale, cn_scale):
    """异步执行草图引导图像生成。"""
    try:
        _set_status(task_id, "running")

        sketch_image = _decode_base64_image(sketch_base64)
        canny_image = _build_canny_image(sketch_image)

        pipeline = get_pipeline()
        result = pipeline(
            prompt=_build_prompt(prompt),
            negative_prompt=config["inference"]["negative_prompt"],
            image=canny_image,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            controlnet_conditioning_scale=cn_scale,
            control_guidance_start=config["inference"]["control_guidance_start"],
            control_guidance_end=config["inference"]["control_guidance_end"],
        )
        result_image = result.images[0]
        result_base64 = _image_to_base64(result_image)

        redis_client.set(
            f"task:{task_id}:result",
            result_base64,
            ex=config["redis"]["ttl"],
        )
        _set_status(task_id, "done")
        return {"task_id": task_id, "status": "done"}
    except Exception as exc:
        _set_status(task_id, "failed")
        redis_client.set(
            f"task:{task_id}:error",
            str(exc),
            ex=config["redis"]["ttl"],
        )
        return {"task_id": task_id, "status": "failed", "error": str(exc)}

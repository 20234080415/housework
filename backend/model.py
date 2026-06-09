import logging
from pathlib import Path

import torch
import yaml
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)


BASE_DIR = Path(__file__).resolve().parents[1]
_pipeline = None
logger = logging.getLogger(__name__)


def load_config():
    """读取项目配置。"""
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_pipeline():
    """获取全局单例推理管线。"""
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    config = load_config()
    model_config = config["models"]

    # 加载 ControlNet 模型。
    controlnet = ControlNetModel.from_pretrained(
        model_config["controlnet_path"],
        torch_dtype=torch.float16,
    )

    # 加载 Stable Diffusion + ControlNet 推理管线，并关闭安全检查。
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        model_config["sd_path"],
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    # 根据配置选择采样调度器。
    scheduler_name = config["inference"]["scheduler"]
    if scheduler_name == "dpmpp_2m_karras":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
    elif scheduler_name == "unipc":
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    else:
        raise ValueError(f"不支持的调度器：{scheduler_name}")

    # 开启显存优化能力。
    pipeline.enable_model_cpu_offload()
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except Exception as exc:
        logger.warning("xFormers 不可用，已跳过显存高效注意力：%s", exc)

    _pipeline = pipeline
    return _pipeline

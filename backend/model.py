from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parents[1]


def load_config():
    """读取模型配置。"""
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


class SketchToImageModel:
    """草图生成图像模型占位类。"""

    def __init__(self):
        self.config = load_config()

    def generate(self, prompt, sketch_path):
        """后续接入 diffusers 与 ControlNet 推理。"""
        raise NotImplementedError("模型推理将在后续任务中实现")

import gc
import json
import math
import time
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image
from scipy import linalg
from skimage.metrics import structural_similarity
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import Inception_V3_Weights, inception_v3
from transformers import CLIPModel, CLIPProcessor


matplotlib.rcParams["font.family"] = "DejaVu Sans"


def load_config():
    """读取项目统一配置。"""
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def list_image_files(directory):
    """过滤系统文件和 Notebook 临时文件。"""
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in suffixes
    )


def select_samples(config):
    """按固定类别和索引选择可复现的草图-照片配对。"""
    experiment = config["experiment"]
    dataset_root = Path(experiment["dataset_root"])
    sketch_root = dataset_root / "sketch" / "tx_000000000000"
    photo_root = dataset_root / "photo" / "tx_000000000000"
    sample_index = int(experiment["sample_index"])
    samples = []

    for category in experiment["categories"]:
        sketch_files = list_image_files(sketch_root / category)
        if not sketch_files:
            raise FileNotFoundError(f"类别缺少草图：{category}")
        sketch_path = sketch_files[sample_index % len(sketch_files)]
        photo_id = sketch_path.stem.rsplit("-", 1)[0]
        photo_path = photo_root / category / f"{photo_id}.jpg"
        if not photo_path.exists():
            raise FileNotFoundError(f"草图缺少配对照片：{sketch_path}")
        samples.append(
            {
                "category": category,
                "label": category.replace("_", " ").replace("(", "").replace(")", ""),
                "sketch_path": sketch_path,
                "photo_path": photo_path,
            }
        )
    return samples


def prepare_condition(sketch_path, width, height, canny_config):
    """将草图转换为 ControlNet 使用的三通道 Canny 条件图。"""
    sketch = np.array(Image.open(sketch_path).convert("RGB").resize((width, height)))
    gray = cv2.cvtColor(sketch, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(
        gray,
        int(canny_config["low_threshold"]),
        int(canny_config["high_threshold"]),
    )
    return Image.fromarray(edges).convert("RGB")


def clear_cuda():
    """释放上一阶段模型占用的显存和内存。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def configure_pipeline(pipe):
    """应用统一调度器和 8GB 显存优化。"""
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing("max")
    pipe.enable_vae_slicing()
    return pipe


def build_prompt(label, inference_config):
    """构建与后端一致的正向提示词。"""
    return f"a realistic photo of a {label}, {inference_config['prompt_suffix']}"


def generate_baseline(config, samples, output_dir):
    """运行不使用结构条件的 Stable Diffusion 文本基线。"""
    model_path = config["models"]["sd_path"]
    experiment = config["experiment"]
    inference = config["inference"]
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        low_cpu_mem_usage=True,
    )
    pipe = configure_pipeline(pipe)
    records = []

    for index, sample in enumerate(samples):
        generator = torch.Generator(device="cpu").manual_seed(
            int(experiment["seed"]) + index
        )
        prompt = build_prompt(sample["label"], inference)
        start = time.perf_counter()
        image = pipe(
            prompt=prompt,
            negative_prompt=inference["negative_prompt"],
            num_inference_steps=int(experiment["inference_steps"]),
            guidance_scale=float(inference["default_cfg"]),
            width=int(experiment["width"]),
            height=int(experiment["height"]),
            generator=generator,
        ).images[0]
        latency = time.perf_counter() - start
        image_path = output_dir / "baseline" / f"{sample['category']}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)
        records.append({**sample, "image_path": image_path, "latency": latency})
        print(f"基线完成：{sample['category']}，{latency:.2f} 秒")

    del pipe
    clear_cuda()
    return records


def load_controlnet_pipeline(config):
    """加载本地 Stable Diffusion 与 ControlNet-Canny。"""
    controlnet = ControlNetModel.from_pretrained(
        config["models"]["controlnet_path"],
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config["models"]["sd_path"],
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        low_cpu_mem_usage=True,
    )
    return configure_pipeline(pipe)


def generate_controlnet(config, samples, output_dir):
    """运行主方法并保留管线用于消融实验。"""
    experiment = config["experiment"]
    inference = config["inference"]
    pipe = load_controlnet_pipeline(config)
    records = []

    for index, sample in enumerate(samples):
        condition = prepare_condition(
            sample["sketch_path"],
            int(experiment["width"]),
            int(experiment["height"]),
            config["canny"],
        )
        generator = torch.Generator(device="cpu").manual_seed(
            int(experiment["seed"]) + index
        )
        prompt = build_prompt(sample["label"], inference)
        start = time.perf_counter()
        image = pipe(
            prompt=prompt,
            negative_prompt=inference["negative_prompt"],
            image=condition,
            num_inference_steps=int(experiment["inference_steps"]),
            guidance_scale=float(inference["default_cfg"]),
            controlnet_conditioning_scale=float(
                experiment["main_control_scale"]
            ),
            control_guidance_start=float(inference["control_guidance_start"]),
            control_guidance_end=float(inference["control_guidance_end"]),
            generator=generator,
        ).images[0]
        latency = time.perf_counter() - start
        image_path = output_dir / "controlnet" / f"{sample['category']}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)
        records.append(
            {
                **sample,
                "condition_path": output_dir
                / "conditions"
                / f"{sample['category']}.png",
                "image_path": image_path,
                "latency": latency,
            }
        )
        records[-1]["condition_path"].parent.mkdir(parents=True, exist_ok=True)
        condition.save(records[-1]["condition_path"])
        print(f"ControlNet 完成：{sample['category']}，{latency:.2f} 秒")
    return pipe, records


def run_ablation(config, pipe, sample, output_dir):
    """在同一草图上运行不同 ControlNet 强度的消融实验。"""
    experiment = config["experiment"]
    inference = config["inference"]
    condition = prepare_condition(
        sample["sketch_path"],
        int(experiment["width"]),
        int(experiment["height"]),
        config["canny"],
    )
    records = []

    for scale in experiment["ablation_scales"]:
        generator = torch.Generator(device="cpu").manual_seed(int(experiment["seed"]))
        start = time.perf_counter()
        image = pipe(
            prompt=build_prompt(sample["label"], inference),
            negative_prompt=inference["negative_prompt"],
            image=condition,
            num_inference_steps=int(experiment["inference_steps"]),
            guidance_scale=float(inference["default_cfg"]),
            controlnet_conditioning_scale=float(scale),
            control_guidance_start=float(inference["control_guidance_start"]),
            control_guidance_end=float(inference["control_guidance_end"]),
            generator=generator,
        ).images[0]
        latency = time.perf_counter() - start
        image_path = output_dir / "ablation" / f"scale_{float(scale):.1f}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)
        records.append(
            {
                "scale": float(scale),
                "image_path": image_path,
                "condition": condition,
                "latency": latency,
                "label": sample["label"],
            }
        )
        print(f"消融完成：scale={float(scale):.1f}，{latency:.2f} 秒")
    return records


def edge_metrics(condition, generated, canny_config):
    """计算输入边缘与生成图边缘之间的结构一致性。"""
    condition_gray = np.array(condition.convert("L"))
    generated_array = np.array(generated.convert("RGB"))
    generated_gray = cv2.cvtColor(generated_array, cv2.COLOR_RGB2GRAY)
    generated_edge = cv2.Canny(
        generated_gray,
        int(canny_config["low_threshold"]),
        int(canny_config["high_threshold"]),
    )
    condition_binary = condition_gray > 0
    generated_binary = generated_edge > 0
    intersection = np.logical_and(condition_binary, generated_binary).sum()
    precision = intersection / max(generated_binary.sum(), 1)
    recall = intersection / max(condition_binary.sum(), 1)
    edge_f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    edge_ssim = structural_similarity(condition_gray, generated_edge, data_range=255)
    return float(edge_ssim), float(edge_f1)


def calculate_clip_scores(config, records):
    """使用 CLIP 计算提示词与生成图的语义一致性。"""
    model_name = config["experiment"]["clip_model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    images = [Image.open(record["image_path"]).convert("RGB") for record in records]
    texts = [
        build_prompt(record["label"], config["inference"]) for record in records
    ]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        image_features = outputs.image_embeds / outputs.image_embeds.norm(
            dim=-1, keepdim=True
        )
        text_features = outputs.text_embeds / outputs.text_embeds.norm(
            dim=-1, keepdim=True
        )
        scores = (image_features * text_features).sum(dim=-1) * 100
    result = scores.detach().cpu().tolist()
    del model
    clear_cuda()
    return [float(score) for score in result]


def inception_features(image_paths, extractor, preprocess, device):
    """提取 InceptionV3 早期层的 64 维特征用于探索性 FID。"""
    features = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feature_map = extractor(tensor)["features"]
            pooled = feature_map.mean(dim=(2, 3))
            features.append(pooled.detach().cpu().numpy())
    return np.concatenate(features, axis=0)


def frechet_distance(features_a, features_b):
    """计算带数值稳定项的 Fréchet 距离。"""
    mean_a, mean_b = features_a.mean(axis=0), features_b.mean(axis=0)
    cov_a = np.cov(features_a, rowvar=False)
    cov_b = np.cov(features_b, rowvar=False)
    epsilon = 1e-6
    identity = np.eye(cov_a.shape[0]) * epsilon
    covariance_mean = linalg.sqrtm((cov_a + identity) @ (cov_b + identity))
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    difference = mean_a - mean_b
    value = (
        difference.dot(difference)
        + np.trace(cov_a)
        + np.trace(cov_b)
        - 2 * np.trace(covariance_mean)
    )
    return float(max(value, 0.0))


def calculate_exploratory_fid(reference_paths, baseline_paths, control_paths):
    """用五组样本计算仅供课程展示的探索性 FID。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights).to(device).eval()
    extractor = create_feature_extractor(model, return_nodes={"maxpool1": "features"})
    preprocess = weights.transforms()
    reference_features = inception_features(
        reference_paths, extractor, preprocess, device
    )
    baseline_features = inception_features(
        baseline_paths, extractor, preprocess, device
    )
    control_features = inception_features(
        control_paths, extractor, preprocess, device
    )
    baseline_fid = frechet_distance(reference_features, baseline_features)
    control_fid = frechet_distance(reference_features, control_features)
    del extractor, model
    clear_cuda()
    return baseline_fid, control_fid


def save_grids(samples, baseline_records, control_records, output_dir):
    """保存草图、参考图、基线和 ControlNet 的对比网格。"""
    figure, axes = plt.subplots(len(samples), 4, figsize=(13, 3 * len(samples)))
    for row, (sample, baseline, control) in enumerate(
        zip(samples, baseline_records, control_records)
    ):
        images = [
            Image.open(sample["sketch_path"]).convert("RGB"),
            Image.open(sample["photo_path"]).convert("RGB"),
            Image.open(baseline["image_path"]).convert("RGB"),
            Image.open(control["image_path"]).convert("RGB"),
        ]
        titles = ["Sketch", "Reference", "SD baseline", "ControlNet"]
        for column, (image, title) in enumerate(zip(images, titles)):
            axes[row, column].imshow(image)
            axes[row, column].set_title(
                f"{sample['label']} - {title}" if column == 0 else title
            )
            axes[row, column].axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "comparison_grid.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_ablation_plot(ablation_records, output_dir):
    """保存消融样本网格和指标曲线。"""
    figure, axes = plt.subplots(1, len(ablation_records), figsize=(18, 4))
    for axis, record in zip(axes, ablation_records):
        axis.imshow(Image.open(record["image_path"]).convert("RGB"))
        axis.set_title(f"scale={record['scale']:.1f}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "ablation_grid.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    scales = [record["scale"] for record in ablation_records]
    axis.plot(
        scales,
        [record["edge_ssim"] for record in ablation_records],
        marker="o",
        label="Edge SSIM",
    )
    axis.plot(
        scales,
        [record["edge_f1"] for record in ablation_records],
        marker="s",
        label="Edge F1",
    )
    axis.set_xlabel("ControlNet conditioning scale")
    axis.set_ylabel("Score")
    axis.set_title("Control strength ablation")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "ablation_metrics.png", dpi=160)
    plt.close(figure)


def main():
    """执行课程设计的完整小样本实验。"""
    config = load_config()
    experiment = config["experiment"]
    output_dir = Path(experiment["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    samples = select_samples(config)

    baseline_records = generate_baseline(config, samples, output_dir)
    pipe, control_records = generate_controlnet(config, samples, output_dir)
    ablation_records = run_ablation(config, pipe, samples[1], output_dir)
    del pipe
    clear_cuda()

    for record in baseline_records:
        condition = prepare_condition(
            record["sketch_path"],
            int(experiment["width"]),
            int(experiment["height"]),
            config["canny"],
        )
        image = Image.open(record["image_path"]).convert("RGB")
        record["edge_ssim"], record["edge_f1"] = edge_metrics(
            condition, image, config["canny"]
        )
    for record in control_records:
        condition = Image.open(record["condition_path"]).convert("RGB")
        image = Image.open(record["image_path"]).convert("RGB")
        record["edge_ssim"], record["edge_f1"] = edge_metrics(
            condition, image, config["canny"]
        )
    for record in ablation_records:
        image = Image.open(record["image_path"]).convert("RGB")
        record["edge_ssim"], record["edge_f1"] = edge_metrics(
            record["condition"], image, config["canny"]
        )

    all_generated = baseline_records + control_records
    clip_scores = calculate_clip_scores(config, all_generated)
    for record, score in zip(all_generated, clip_scores):
        record["clip_score"] = score

    baseline_fid, control_fid = calculate_exploratory_fid(
        [sample["photo_path"] for sample in samples],
        [record["image_path"] for record in baseline_records],
        [record["image_path"] for record in control_records],
    )

    summary = pd.DataFrame(
        [
            {
                "方法": "Stable Diffusion baseline",
                "探索性 FID ↓": baseline_fid,
                "边缘 SSIM ↑": np.mean(
                    [record["edge_ssim"] for record in baseline_records]
                ),
                "边缘 F1 ↑": np.mean(
                    [record["edge_f1"] for record in baseline_records]
                ),
                "CLIP Score ↑": np.mean(
                    [record["clip_score"] for record in baseline_records]
                ),
                "推理延迟(s) ↓": np.mean(
                    [record["latency"] for record in baseline_records]
                ),
            },
            {
                "方法": "ControlNet-Canny",
                "探索性 FID ↓": control_fid,
                "边缘 SSIM ↑": np.mean(
                    [record["edge_ssim"] for record in control_records]
                ),
                "边缘 F1 ↑": np.mean(
                    [record["edge_f1"] for record in control_records]
                ),
                "CLIP Score ↑": np.mean(
                    [record["clip_score"] for record in control_records]
                ),
                "推理延迟(s) ↓": np.mean(
                    [record["latency"] for record in control_records]
                ),
            },
        ]
    )
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")

    ablation_frame = pd.DataFrame(
        [
            {
                "control_scale": record["scale"],
                "edge_ssim": record["edge_ssim"],
                "edge_f1": record["edge_f1"],
                "latency_s": record["latency"],
            }
            for record in ablation_records
        ]
    )
    ablation_frame.to_csv(
        output_dir / "ablation.csv", index=False, encoding="utf-8-sig"
    )
    save_grids(samples, baseline_records, control_records, output_dir)
    save_ablation_plot(ablation_records, output_dir)

    details = {
        "dataset": {
            "name": "Sketchy Database",
            "categories": len(
                [
                    path
                    for path in (
                        Path(experiment["dataset_root"])
                        / "sketch"
                        / "tx_000000000000"
                    ).iterdir()
                    if path.is_dir() and not path.name.startswith(".")
                ]
            ),
            "sketches": 75481,
            "photos": 12500,
            "evaluation_samples": len(samples),
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "diffusers_steps": int(experiment["inference_steps"]),
            "seed": int(experiment["seed"]),
        },
        "summary": json.loads(summary.to_json(orient="records", force_ascii=False)),
        "ablation": json.loads(
            ablation_frame.to_json(orient="records", force_ascii=False)
        ),
        "limitation": "FID 仅基于 5 个配对样本，用于课程实验流程展示，不具备大样本统计意义。",
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(details, file, ensure_ascii=False, indent=2)
    print(summary.to_string(index=False))
    print(f"实验结果已保存到：{output_dir}")


if __name__ == "__main__":
    main()

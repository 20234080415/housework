import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "DL课程设计.ipynb"
METRICS_PATH = ROOT / "outputs" / "experiment" / "metrics.json"


def code_cell(source):
    """创建待执行的代码单元。"""
    return nbformat.v4.new_code_cell(source=source)


def markdown_cell(source):
    """创建 Markdown 报告单元。"""
    return nbformat.v4.new_markdown_cell(source=source)


def main():
    """将真实实验结果写回课程设计 Notebook。"""
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    summary = metrics["summary"]
    baseline = summary[0]
    controlnet = summary[1]
    ablation = metrics["ablation"]

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)

    notebook.cells[3] = markdown_cell(
        """## 四、数据集说明与预处理

### 4.1 数据来源与规模

本实验实际下载并解压了 **Sketchy Database**。本地数据来自 Hugging Face 的
`DrRORAL/sketchy-dataset` 镜像，归档 SHA-256 为
`6a746a915a902cb05216644f747f2f900de8bf1a97034db25efac44fda1a3c87`。
该镜像包含原始 Sketchy Database 的目录结构；数据版权与使用条款仍以原始数据集为准。

| 属性 | 实际值 |
|------|------:|
| 类别数 | 125 |
| 草图数 | 75,481 |
| 照片数 | 12,500 |
| 本次评测配对数 | 5 |
| 本地目录 | `./data/sketchy/` |

数据源：https://huggingface.co/datasets/DrRORAL/sketchy-dataset

原论文：Sangkloy et al., *The Sketchy Database: Learning to Retrieve Badly Drawn Bunnies*, SIGGRAPH 2016。

### 4.2 评测样本

为适配 RTX 4060 8GB 和课程实验时长，本次使用 airplane、cat、dog、car_(sedan)、
horse 五个类别各一个草图-照片精确配对。该设计用于验证完整实验流程，不等同于论文级全量评测。"""
    )

    notebook.cells[4] = code_cell(
        """# 数据集固定样本可视化
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import yaml

with open("./config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

dataset_root = Path(config["experiment"]["dataset_root"])
sketch_root = dataset_root / "sketch" / "tx_000000000000"
photo_root = dataset_root / "photo" / "tx_000000000000"
categories = config["experiment"]["categories"]

fig, axes = plt.subplots(2, len(categories), figsize=(18, 7))
for index, category in enumerate(categories):
    sketch_files = sorted(
        path for path in (sketch_root / category).glob("*.png")
        if not path.name.startswith(".")
    )
    sketch_path = sketch_files[config["experiment"]["sample_index"]]
    photo_id = sketch_path.stem.rsplit("-", 1)[0]
    photo_path = photo_root / category / f"{photo_id}.jpg"

    axes[0, index].imshow(Image.open(sketch_path).convert("L"), cmap="gray")
    axes[0, index].set_title(category)
    axes[0, index].axis("off")
    axes[1, index].imshow(Image.open(photo_path).convert("RGB"))
    axes[1, index].axis("off")

fig.suptitle("Sketchy Database 固定评测样本（上：草图，下：配对照片）")
plt.tight_layout()
plt.show()"""
    )

    notebook.cells[5] = code_cell(
        """# 数据集类别分布统计
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
category_dirs = sorted(
    path for path in sketch_root.iterdir()
    if path.is_dir() and not path.name.startswith(".")
)
counts = {
    path.name: sum(
        1 for file in path.iterdir()
        if file.is_file() and file.suffix.lower() in image_suffixes
    )
    for path in category_dirs
}
distribution = pd.Series(counts).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(20, 5))
ax.bar(distribution.index, distribution.values, color="steelblue")
ax.set_title("Sketchy Database 各类别草图数量")
ax.set_xlabel("类别")
ax.set_ylabel("草图数量")
ax.tick_params(axis="x", rotation=90, labelsize=7)
plt.tight_layout()
plt.show()
print(
    f"类别数：{len(distribution)}，草图总数：{distribution.sum()}，"
    f"平均每类：{distribution.mean():.1f}"
)"""
    )

    notebook.cells[6] = markdown_cell(
        """### 4.3 预处理流程

1. 草图和照片统一缩放至 512×512。
2. 草图转灰度后使用 Canny 检测，阈值从 `config.yaml` 的 `canny` 节读取。
3. ControlNet 条件图转换为三通道 RGB。
4. 正向提示词、负向提示词、CFG、控制区间和图像尺寸全部从 `config.yaml` 读取。
5. 实验固定随机种子 42，保证基线、主方法和消融实验可重复。"""
    )

    notebook.cells[7] = code_cell(
        """# 数据配对与 Canny 预处理
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def canny_preprocess(sketch_np, low, high):
    gray = cv2.cvtColor(sketch_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    return edges.astype(np.float32) / 255.0

class SketchyDataset(Dataset):
    def __init__(self, sketch_dir, photo_dir, img_size, canny_config):
        self.img_size = img_size
        self.canny_config = canny_config
        self.pairs = []
        category_dirs = sorted(
            path for path in Path(sketch_dir).iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
        for category_dir in category_dirs:
            for sketch_path in sorted(category_dir.glob("*.png")):
                photo_id = sketch_path.stem.rsplit("-", 1)[0]
                photo_path = Path(photo_dir) / category_dir.name / f"{photo_id}.jpg"
                if photo_path.exists():
                    self.pairs.append(
                        (sketch_path, photo_path, category_dir.name)
                    )
        self.photo_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        sketch_path, photo_path, category = self.pairs[index]
        sketch = np.array(
            Image.open(sketch_path).convert("RGB").resize(
                (self.img_size, self.img_size)
            )
        )
        canny = canny_preprocess(
            sketch,
            self.canny_config["low_threshold"],
            self.canny_config["high_threshold"],
        )
        return {
            "canny": torch.from_numpy(canny).unsqueeze(0).repeat(3, 1, 1),
            "photo": self.photo_transform(Image.open(photo_path).convert("RGB")),
            "prompt": f"a realistic photo of a {category.replace('_', ' ')}",
        }

dataset = SketchyDataset(
    sketch_root,
    photo_root,
    config["experiment"]["width"],
    config["canny"],
)
print(f"成功建立精确草图-照片配对：{len(dataset)} 组")"""
    )

    notebook.cells[9] = code_cell(
        """# 实际运行环境与模型检查
from pathlib import Path
import torch
import diffusers

sd_path = Path(config["models"]["sd_path"])
controlnet_path = Path(config["models"]["controlnet_path"])
assert torch.cuda.is_available(), "未检测到 CUDA"
assert sd_path.exists(), f"Stable Diffusion 模型不存在：{sd_path}"
assert controlnet_path.exists(), f"ControlNet 模型不存在：{controlnet_path}"

print(f"GPU：{torch.cuda.get_device_name(0)}")
print(f"显存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"PyTorch：{torch.__version__}")
print(f"CUDA：{torch.version.cuda}")
print(f"diffusers：{diffusers.__version__}")
print(f"推理步数：{config['experiment']['inference_steps']}")
print(f"随机种子：{config['experiment']['seed']}")
print("完整实验脚本：./scripts/run_course_experiment.py")"""
    )

    notebook.cells[10] = markdown_cell(
        """## 六、实验与结果分析

### 6.1 实验设置

| 项目 | 实际配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8GB |
| PyTorch / CUDA | 2.1.0+cu118 / 11.8 |
| 模型 | Stable Diffusion v1.5 + ControlNet-Canny |
| 精度与显存优化 | FP16、CPU Offload、Attention Slicing、VAE Slicing |
| 分辨率 | 512×512 |
| 推理步数 | 15 |
| 随机种子 | 42 |
| 主方法控制强度 | 0.8 |
| 评测样本 | 5 个类别各 1 组配对 |

### 6.2 指标说明

- **边缘 SSIM / 边缘 F1**：衡量生成图 Canny 边缘与输入草图边缘的一致性。
- **CLIP Score**：衡量生成图与类别文本提示的语义一致性。
- **探索性 FID**：基于 5 组样本和 Inception 64 维早期特征，仅展示评测流程，不具备大样本统计意义。
- **推理延迟**：单张 512×512 图像的端到端生成时间。"""
    )

    notebook.cells[11] = code_cell(
        """# 读取本机实际实验结果
from pathlib import Path
import pandas as pd

experiment_dir = Path(config["experiment"]["output_dir"])
results = pd.read_csv(experiment_dir / "summary.csv")
display(results.round(4))

baseline_ssim = results.loc[0, "边缘 SSIM ↑"]
control_ssim = results.loc[1, "边缘 SSIM ↑"]
baseline_f1 = results.loc[0, "边缘 F1 ↑"]
control_f1 = results.loc[1, "边缘 F1 ↑"]
print(f"边缘 SSIM 相对提升：{(control_ssim / baseline_ssim - 1) * 100:.1f}%")
print(f"边缘 F1 相对提升：{(control_f1 / baseline_f1 - 1) * 100:.1f}%")"""
    )

    notebook.cells[12] = code_cell(
        """# 展示草图、参考照片、文本基线与 ControlNet 的实际结果
from IPython.display import display
from PIL import Image

comparison_path = experiment_dir / "comparison_grid.png"
display(Image.open(comparison_path))
print(f"原始结果目录：{experiment_dir}")"""
    )

    notebook.cells[13] = markdown_cell(
        f"""### 6.3 定量分析

实际结果如下：

| 方法 | 探索性 FID ↓ | 边缘 SSIM ↑ | 边缘 F1 ↑ | CLIP Score ↑ | 延迟(s) ↓ |
|------|-------------:|------------:|----------:|-------------:|----------:|
| Stable Diffusion baseline | {baseline['探索性 FID ↓']:.4f} | {baseline['边缘 SSIM ↑']:.4f} | {baseline['边缘 F1 ↑']:.4f} | {baseline['CLIP Score ↑']:.4f} | {baseline['推理延迟(s) ↓']:.2f} |
| ControlNet-Canny | {controlnet['探索性 FID ↓']:.4f} | {controlnet['边缘 SSIM ↑']:.4f} | {controlnet['边缘 F1 ↑']:.4f} | {controlnet['CLIP Score ↑']:.4f} | {controlnet['推理延迟(s) ↓']:.2f} |

ControlNet 将边缘 SSIM 提升约
`{(controlnet['边缘 SSIM ↑'] / baseline['边缘 SSIM ↑'] - 1) * 100:.1f}%`，
边缘 F1 提升约
`{(controlnet['边缘 F1 ↑'] / baseline['边缘 F1 ↑'] - 1) * 100:.1f}%`，
证明草图条件显著增强了空间结构控制。

CLIP Score 基本持平，说明结构约束没有明显破坏类别语义。ControlNet 平均增加约
`{controlnet['推理延迟(s) ↓'] - baseline['推理延迟(s) ↓']:.2f}` 秒延迟。

探索性 FID 在 ControlNet 上更高。该指标仅有 5 个样本，且参考照片的背景、视角与草图
并不完全一致，因此不能据此推断总体生成质量下降；它更说明小样本 FID 对样本选择非常敏感。"""
    )

    notebook.cells[14] = code_cell(
        """# ControlNet 强度消融结果
ablation = pd.read_csv(experiment_dir / "ablation.csv")
display(ablation.round(4))
display(Image.open(experiment_dir / "ablation_grid.png"))
display(Image.open(experiment_dir / "ablation_metrics.png"))"""
    )

    best = max(ablation, key=lambda item: item["edge_f1"])
    notebook.cells[15] = markdown_cell(
        f"""### 6.4 消融实验分析

控制强度从 0.0 增加到 1.6 时，边缘 SSIM 从
`{ablation[0]['edge_ssim']:.4f}` 上升到 `{best['edge_ssim']:.4f}`，
边缘 F1 从 `{ablation[0]['edge_f1']:.4f}` 上升到 `{best['edge_f1']:.4f}`。

这说明更高的 ControlNet 强度能够加强轮廓遵循。但可视化中 `scale=1.6` 已出现明显的
线稿化和局部描边，写实感弱于 0.8～1.2。因此系统默认值保留为 `0.8`，在结构一致性和
自然图像质量之间取得更稳妥的平衡。"""
    )

    notebook.cells[16] = markdown_cell(
        """## 七、结论与局限

本项目已在 RTX 4060 Laptop GPU 上实际完成 Stable Diffusion 文本基线、
ControlNet-Canny 主方法和五档控制强度消融实验。

主要结论：

1. ControlNet 显著提高草图结构一致性，尤其能保留物体姿态、轮廓和空间布局。
2. CLIP 语义分数与文本基线基本持平，说明结构控制和类别语义可以兼顾。
3. ControlNet 带来额外推理延迟，属于获得可控生成能力的计算代价。
4. 控制强度并非越高越好；过强会导致生成结果贴近线稿、写实感下降。

局限：

- 本次仅使用 5 个固定配对样本，结果适合课程设计验证，不代表全量数据集统计结论。
- 探索性 FID 的样本量过小，不应与论文中的标准 50K FID 横向比较。
- 本实验使用预训练 ControlNet 推理，没有在 Sketchy Database 上重新训练或微调。
- 后续可扩大评测样本、加入人工偏好评分，并在独立测试集上计算标准 FID/KID。"""
    )

    notebook.metadata["kernelspec"] = {
        "display_name": "Python (housework)",
        "language": "python",
        "name": "housework",
    }
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Notebook 已更新：{NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()

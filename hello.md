确认一下方向，方便给你最贴合的步骤：Q: 步骤指导的覆盖范围？
A: 完整走一遍（从环境到前端全覆盖）

Q: 你的开发环境是？
A: 混合：训练用云，本地开发

Q: 指导风格偏好？
A: 直接给我代码框架（能跑就行）

好，开始输出完整指导。内容比较长，我会分成几个清晰的阶段，每阶段给你可直接运行的代码框架。

---

## 总体项目结构

```
controlnet-sketch/
├── inference/          # 推理核心（云端跑）
│   ├── pipeline.py
│   └── preprocess.py
├── train/              # LoRA 微调（云端跑）
│   └── train_lora.sh
├── backend/            # FastAPI 后端（本地/云端）
│   ├── main.py
│   ├── tasks.py
│   └── requirements.txt
├── frontend/           # React 前端（本地开发）
│   └── src/
│       ├── App.jsx
│       └── SketchCanvas.jsx
└── docker-compose.yml
```

---

## Phase 1｜环境搭建（云端）

SSH 进云服务器后：

```bash
# 1. 创建 conda 环境
conda create -n controlnet python=3.10 -y
conda activate controlnet

# 2. 安装 PyTorch（CUDA 12.1）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 diffusers 生态
pip install diffusers transformers accelerate peft xformers
pip install opencv-python controlnet-aux

# 4. 验证
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Phase 2｜推理核心

**`inference/preprocess.py`** — 草图预处理：

```python
import cv2
import numpy as np
from PIL import Image
from controlnet_aux import CannyDetector, HEDdetector

canny = CannyDetector()
hed = HEDdetector.from_pretrained("lllyasviel/Annotators")

def preprocess_sketch(image: Image.Image, mode: str = "canny") -> Image.Image:
    """
    将用户手绘草图转化为 ControlNet 控制图
    mode: 'canny' | 'hed'
    """
    img = image.convert("RGB").resize((512, 512))
    
    if mode == "canny":
        # 手绘草图先模糊再提边缘，避免噪声
        arr = np.array(img)
        blurred = cv2.GaussianBlur(arr, (5, 5), 0)
        control = canny(Image.fromarray(blurred), low_threshold=50, high_threshold=150)
    elif mode == "hed":
        control = hed(img)
    
    return control  # PIL Image, 单通道边缘图
```

**`inference/pipeline.py`** — 推理主逻辑：

```python
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from PIL import Image
from .preprocess import preprocess_sketch

# 全局加载，避免每次请求重新加载（冷启动只跑一次）
_pipe = None

def get_pipeline(lora_path: str = None):
    global _pipe
    if _pipe is not None:
        return _pipe
    
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to("cuda")
    
    if lora_path:
        pipe.load_lora_weights(lora_path)
    
    _pipe = pipe
    return pipe

def generate(
    sketch: Image.Image,
    prompt: str,
    negative_prompt: str = "blurry, low quality, distorted",
    control_mode: str = "canny",
    num_steps: int = 20,
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 1.0,
    seed: int = 42,
    lora_path: str = None,
) -> Image.Image:
    pipe = get_pipeline(lora_path)
    control_image = preprocess_sketch(sketch, mode=control_mode)
    
    generator = torch.manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator=generator,
    )
    return result.images[0]  # PIL Image
```

快速验证推理是否跑通：

```python
# test_inference.py
from PIL import Image
from inference.pipeline import generate

sketch = Image.new("RGB", (512, 512), "white")  # 空白草图测试
result = generate(sketch, prompt="a cat sitting on a chair, realistic")
result.save("test_output.png")
print("OK")
```

---

## Phase 3｜LoRA 微调

**数据准备：**

```
train_data/
├── images/          # 50~200 张目标风格图（512x512）
└── metadata.jsonl   # 每行: {"file_name": "001.jpg", "text": "anime style illustration"}
```

**`train/train_lora.sh`：**

```bash
#!/bin/bash
accelerate launch diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="./train_data" \
  --output_dir="./lora_weights/anime_style" \
  --resolution=512 \
  --train_batch_size=4 \
  --num_train_epochs=20 \
  --max_train_steps=2000 \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine" \
  --rank=8 \
  --seed=42 \
  --checkpointing_steps=500 \
  --mixed_precision="fp16"
```

训练完成后 `lora_weights/anime_style/` 下会有 `.safetensors` 权重文件，传回本地或直接在云端推理服务中挂载。

---

## Phase 4｜后端（FastAPI + Celery）

**`backend/requirements.txt`：**

```
fastapi
uvicorn[standard]
celery[redis]
redis
python-multipart
pillow
```

**`backend/tasks.py`** — Celery 异步任务：

```python
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from celery import Celery
from PIL import Image
import base64, io
from inference.pipeline import generate

app = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def base64_to_pil(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data))

@app.task(bind=True)
def generate_task(self, sketch_b64: str, params: dict):
    try:
        self.update_state(state="PROGRESS", meta={"status": "preprocessing"})
        sketch = base64_to_pil(sketch_b64)
        
        self.update_state(state="PROGRESS", meta={"status": "generating"})
        result = generate(sketch, **params)
        
        return {"status": "done", "image": pil_to_base64(result)}
    except Exception as e:
        self.update_state(state="FAILURE", meta={"status": str(e)})
        raise
```

**`backend/main.py`** — FastAPI 接口：

```python
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from .tasks import generate_task, app as celery_app
import base64, io
from PIL import Image

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/generate")
async def submit_generate(
    sketch: UploadFile,
    prompt: str = Form(...),
    negative_prompt: str = Form("blurry, low quality"),
    num_steps: int = Form(20),
    guidance_scale: float = Form(7.5),
    controlnet_scale: float = Form(1.0),
    control_mode: str = Form("canny"),
    seed: int = Form(42),
    lora_path: str = Form(None),
):
    img_bytes = await sketch.read()
    sketch_b64 = base64.b64encode(img_bytes).decode()
    
    params = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_scale,
        control_mode=control_mode,
        seed=seed,
        lora_path=lora_path,
    )
    task = generate_task.delay(sketch_b64, params)
    return {"task_id": task.id}

@app.get("/result/{task_id}")
def get_result(task_id: str):
    task = celery_app.AsyncResult(task_id)
    if task.state == "PROGRESS":
        return {"status": "processing", "info": task.info}
    elif task.state == "SUCCESS":
        return {"status": "done", "image": task.result["image"]}
    elif task.state == "FAILURE":
        return {"status": "failed", "error": str(task.info)}
    return {"status": task.state}
```

启动命令：

```bash
# 启动 Redis（Docker）
docker run -d -p 6379:6379 redis:alpine

# 启动 Celery worker（推理机器上）
celery -A backend.tasks worker --loglevel=info --concurrency=1

# 启动 FastAPI
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## Phase 5｜前端（React）

```bash
npx create-react-app frontend
cd frontend
npm install fabric axios
```

**`src/SketchCanvas.jsx`** — 画板组件：

```jsx
import { useEffect, useRef } from "react";
import { fabric } from "fabric";

export default function SketchCanvas({ onExport }) {
  const canvasRef = useRef(null);
  const fabricRef = useRef(null);

  useEffect(() => {
    const canvas = new fabric.Canvas(canvasRef.current, {
      width: 512, height: 512,
      backgroundColor: "#ffffff",
      isDrawingMode: true,
    });
    canvas.freeDrawingBrush.width = 4;
    canvas.freeDrawingBrush.color = "#000000";
    fabricRef.current = canvas;
    return () => canvas.dispose();
  }, []);

  const exportPNG = () => {
    const dataURL = fabricRef.current.toDataURL({ format: "png" });
    onExport(dataURL);  // 传给父组件
  };

  const clear = () => fabricRef.current.clear().set("backgroundColor", "#fff").renderAll();

  const setBrush = (size) => { fabricRef.current.freeDrawingBrush.width = size; };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <canvas ref={canvasRef} style={{ border: "1px solid #ccc", borderRadius: 8 }} />
      <div style={{ display: "flex", gap: 8 }}>
        {[2, 4, 8, 16].map(s => (
          <button key={s} onClick={() => setBrush(s)}>笔触 {s}px</button>
        ))}
        <button onClick={clear} style={{ marginLeft: "auto" }}>清空</button>
        <button onClick={exportPNG} style={{ background: "#2563eb", color: "#fff", padding: "6px 16px", borderRadius: 6 }}>
          生成图像
        </button>
      </div>
    </div>
  );
}
```

**`src/App.jsx`** — 主页面：

```jsx
import { useState } from "react";
import axios from "axios";
import SketchCanvas from "./SketchCanvas";

const API = "http://your-cloud-ip:8000";

export default function App() {
  const [prompt, setPrompt] = useState("a cat sitting on a sofa, realistic");
  const [steps, setSteps] = useState(20);
  const [cfg, setCfg] = useState(7.5);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("");

  const handleGenerate = async (dataURL) => {
    setStatus("提交任务中...");
    setResult(null);

    // dataURL -> Blob
    const res = await fetch(dataURL);
    const blob = await res.blob();
    
    const form = new FormData();
    form.append("sketch", blob, "sketch.png");
    form.append("prompt", prompt);
    form.append("num_steps", steps);
    form.append("guidance_scale", cfg);

    const { data } = await axios.post(`${API}/generate`, form);
    const taskId = data.task_id;
    
    // 轮询结果
    setStatus("生成中...");
    const poll = setInterval(async () => {
      const { data: r } = await axios.get(`${API}/result/${taskId}`);
      if (r.status === "done") {
        clearInterval(poll);
        setResult(`data:image/png;base64,${r.image}`);
        setStatus("完成！");
      } else if (r.status === "failed") {
        clearInterval(poll);
        setStatus(`失败: ${r.error}`);
      }
    }, 1500);
  };

  return (
    <div style={{ display: "flex", gap: 24, padding: 24, fontFamily: "Arial" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <h2 style={{ margin: 0 }}>草图输入</h2>
        <SketchCanvas onExport={handleGenerate} />
        <textarea
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          rows={3}
          style={{ padding: 8, borderRadius: 6, border: "1px solid #ccc", width: 512 }}
          placeholder="描述你想生成的图像..."
        />
        <div style={{ display: "flex", gap: 16 }}>
          <label>Steps: <input type="range" min={10} max={50} value={steps} onChange={e => setSteps(+e.target.value)} /> {steps}</label>
          <label>CFG: <input type="range" min={1} max={15} step={0.5} value={cfg} onChange={e => setCfg(+e.target.value)} /> {cfg}</label>
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <h2 style={{ margin: 0 }}>生成结果 <span style={{ fontSize: 14, color: "#6b7280" }}>{status}</span></h2>
        <div style={{ width: 512, height: 512, background: "#f3f4f6", borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center" }}>
          {result
            ? <img src={result} alt="result" style={{ width: 512, height: 512, borderRadius: 8 }} />
            : <span style={{ color: "#9ca3af" }}>等待生成...</span>
          }
        </div>
      </div>
    </div>
  );
}
```

---

## Phase 6｜一键部署（docker-compose）

```yaml
# docker-compose.yml
version: "3.9"
services:
  redis:
    image: redis:alpine
    ports: ["6379:6379"]

  worker:
    build: .
    command: celery -A backend.tasks worker --loglevel=info --concurrency=1
    depends_on: [redis]
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  api:
    build: .
    command: uvicorn backend.main:app --host 0.0.0.0 --port 8000
    ports: ["8000:8000"]
    depends_on: [redis, worker]
```

---

## 各阶段验收节点

| 阶段 | 怎么判断跑通了 |
|------|--------------|
| Phase 2 | `test_inference.py` 输出 `test_output.png` 无报错 |
| Phase 3 | LoRA 训练 loss 曲线稳定下降到 0.05 以下 |
| Phase 4 | `curl` 调 `/generate` 拿到 task_id，`/result/{id}` 返回 base64 图片 |
| Phase 5 | 浏览器画草图，点生成，右侧出图 |

---

按这个顺序来，每个阶段跑通再往下。遇到具体报错或卡住的地方直接贴给我，帮你排查。
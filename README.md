## 效果展示

<img width="1919" height="1079" alt="屏幕截图 2026-06-09 134733" src="https://github.com/user-attachments/assets/66f6754a-9c4a-4e8d-ac85-9144d8768fb9" />
<img width="1919" height="1079" alt="屏幕截图 2026-06-09 134908" src="https://github.com/user-attachments/assets/066e1558-7d61-4227-8fac-790ae57b0414" />
<img width="1919" height="1079" alt="屏幕截图 2026-06-09 134717" src="https://github.com/user-attachments/assets/2c12756a-cb13-4b70-b57f-d04b41e9f51f" />


# 草图引导图像生成系统

基于 ControlNet + Stable Diffusion v1.5 的草图引导图像生成系统。前端使用 React + Fabric.js 提供 512×512 画板，后端使用 FastAPI + Celery + Redis 异步执行图像生成任务。

## 技术栈

- 前端：React、Vite、Fabric.js、Axios
- 后端：FastAPI、Celery、Redis
- AI 推理：diffusers、ControlNet、controlnet_aux、PyTorch
- 推荐推理环境：NVIDIA GPU、Python 3.9、CUDA 11.8

## 项目结构

```text
controlnet-sketch/
├── backend/
│   ├── main.py              # FastAPI 接口
│   ├── model.py             # 模型加载模块
│   ├── tasks.py             # Celery 异步任务
│   └── requirements.txt     # 后端依赖
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── Canvas.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── scripts/
│   └── download_models.sh   # AutoDL 模型下载脚本
├── outputs/
│   └── .gitkeep
├── AGENTS.md
├── config.yaml
└── README.md
```

## 配置说明

所有服务地址、端口、Redis、Celery、模型路径和推理参数都从 `config.yaml` 读取。

默认模型路径：

```yaml
models:
  sd_path: /root/models/sd-v1-5
  controlnet_path: /root/models/controlnet-canny
```

默认后端服务：

```yaml
server:
  host: 0.0.0.0
  port: 8000
```

## 后端启动

进入后端目录并安装依赖：

```bash
cd backend
pip install -r requirements.txt
```

启动 Redis：

```bash
redis-server
```

启动 Celery Worker：

```bash
celery -A tasks.celery_app worker --loglevel=info
```

启动 FastAPI：

```bash
python main.py
```

后端默认地址为：

```text
http://localhost:8000
```

## 模型下载

在 AutoDL 云端环境执行：

```bash
bash scripts/download_models.sh
```

脚本会优先使用 ModelScope 下载模型，如果本机没有 `modelscope`，会回退到 `huggingface-cli`。

下载目标：

- Stable Diffusion v1.5：`/root/models/sd-v1-5`
- ControlNet Canny：`/root/models/controlnet-canny`

## 前端启动

进入前端目录并安装依赖：

```bash
cd frontend
npm install
npm run dev
```

如果后端不在默认地址，可以设置环境变量：

```bash
VITE_API_URL=http://localhost:8000 npm run dev
```

## API 接口

### 健康检查

```http
GET /health
```

返回：

```json
{"status":"ok"}
```

### 创建生成任务

```http
POST /api/generate
```

请求体：

```json
{
  "sketch_base64": "data:image/png;base64,...",
  "prompt": "a beautiful house",
  "steps": 20,
  "cfg_scale": 7.5,
  "cn_scale": 1.0
}
```

返回：

```json
{
  "code": 0,
  "data": {
    "task_id": "..."
  },
  "msg": "ok"
}
```

### 查询任务状态

```http
GET /api/status/{task_id}
```

任务完成时返回：

```json
{
  "code": 0,
  "data": {
    "status": "done",
    "result_base64": "..."
  },
  "msg": "ok"
}
```

## 使用流程

1. 在 AutoDL 环境安装后端依赖并下载模型。
2. 启动 Redis、Celery Worker 和 FastAPI。
3. 在本地 Windows 环境启动前端。
4. 在画板中绘制白色草图，输入提示词，点击生成图像。
5. 前端会自动轮询任务状态，完成后显示生成结果。

## 注意事项

- 生成接口为异步接口，前端通过 `task_id` 轮询结果。
- `outputs/`、`models/`、`node_modules/`、构建产物不会提交到 Git。
- 本地 Windows A 卡环境仅建议运行前端和代码编辑，模型推理建议放在 NVIDIA GPU 云端环境。

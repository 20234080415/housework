# Windows 后端推理机启动指南

本文用于在 Windows NVIDIA GPU 推理机上启动 ControlNet 草图生成后端。

后端由三个常驻服务组成：

1. Memurai：提供 Redis 兼容服务，保存任务状态和生成结果。
2. Celery Worker：加载模型并执行 GPU 图像生成任务。
3. FastAPI：接收前端请求并创建异步任务。

三个服务必须同时运行。

## 运行环境

当前项目默认使用：

- 项目目录：`E:\dp_design\housework`
- Python：`C:\Users\Yin\miniconda3\envs\housework\python.exe`
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- CUDA：11.8
- API 端口：`8000`
- Redis/Memurai 端口：`6379`

服务地址、模型路径和推理参数统一配置在 `config.yaml` 中。

## 首次安装

已经完成环境安装时，可以直接跳到“启动服务”。

### 1. 创建 Conda 环境

在 PowerShell 中运行：

```powershell
conda create -n housework python=3.10 -y
conda activate housework
```

### 2. 安装后端依赖

进入项目目录：

```powershell
cd E:\dp_design\housework
```

安装 Windows CUDA 11.8 依赖：

```powershell
pip install -r .\backend\requirements-win-cu118.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

验证 CUDA：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

输出中应包含：

```text
True
NVIDIA GeForce RTX 4060 Laptop GPU
```

### 3. 下载模型

运行：

```powershell
.\scripts\download_models.ps1
```

默认下载到：

```text
E:\dp_design\housework\models\sd-v1-5
E:\dp_design\housework\models\controlnet-canny
```

模型目录必须与 `config.yaml` 中的配置一致：

```yaml
models:
  sd_path: E:/dp_design/housework/models/sd-v1-5
  controlnet_path: E:/dp_design/housework/models/controlnet-canny
```

## 启动服务

在项目根目录打开三个 PowerShell 窗口。三个窗口启动后都要保持运行。

如果 PowerShell 禁止执行脚本，可以先在当前窗口运行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### 窗口 1：启动 Memurai

```powershell
cd E:\dp_design\housework
.\scripts\start_redis.ps1
```

Memurai 默认监听 `6379` 端口。

### 窗口 2：启动 Celery Worker

确认 Memurai 已启动，然后运行：

```powershell
cd E:\dp_design\housework
.\scripts\start_backend_worker.ps1
```

Worker 首次收到生成任务时才会加载 Stable Diffusion 和 ControlNet 模型，因此第一次生成会比较慢。

看到类似以下内容表示 Worker 已就绪：

```text
celery@... ready.
```

### 窗口 3：启动 FastAPI

```powershell
cd E:\dp_design\housework
.\scripts\start_backend_api.ps1
```

看到类似以下内容表示 API 已启动：

```text
Uvicorn running on http://0.0.0.0:8000
```

## 验证服务

浏览器访问：

```text
http://localhost:8000/health
```

正常返回：

```json
{"status":"ok"}
```

API 调试文档：

```text
http://localhost:8000/docs
```

也可以在 PowerShell 中检查：

```powershell
Invoke-RestMethod http://localhost:8000/health
```

## 连接前端

前端和后端在同一台机器时，API 地址使用：

```text
http://localhost:8000
```

前端在其他电脑时，先在后端推理机执行：

```powershell
ipconfig
```

找到推理机的 IPv4 地址，例如 `192.168.1.100`，前端 API 地址设置为：

```text
http://192.168.1.100:8000
```

两台机器需要位于可互相访问的网络中，并允许 Windows 防火墙放行 TCP `8000` 端口。

## 停止服务

在三个服务窗口中分别按：

```text
Ctrl+C
```

建议按以下顺序停止：

1. FastAPI
2. Celery Worker
3. Memurai

## 常见问题

### 无法连接 Redis

错误中出现 `Connection refused` 或无法连接 `localhost:6379` 时，先确认 `start_redis.ps1` 窗口仍在运行。

检查端口：

```powershell
Get-NetTCPConnection -LocalPort 6379 -State Listen
```

### API 可以访问，但任务一直 pending

通常是 Celery Worker 没有启动，或者 Worker 无法连接 Memurai。检查 Worker 窗口中的错误日志。

### CUDA 不可用

运行：

```powershell
C:\Users\Yin\miniconda3\envs\housework\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

当前环境应输出带有 `cu118` 的 PyTorch 版本，并且 CUDA 状态为 `True`。

### 模型加载失败

检查 `config.yaml` 中的模型路径，并确认以下文件存在：

```text
models\sd-v1-5\model_index.json
models\controlnet-canny\config.json
```

### 显存不足

当前代码已启用模型 CPU Offload。仍然显存不足时，可以关闭占用 GPU 的其他程序，并适当减少推理步数。所有默认推理参数应在 `config.yaml` 中调整。

## 每次启动速查

在三个 PowerShell 窗口中依次运行：

```powershell
.\scripts\start_redis.ps1
```

```powershell
.\scripts\start_backend_worker.ps1
```

```powershell
.\scripts\start_backend_api.ps1
```

最后访问：

```text
http://localhost:8000/health
```

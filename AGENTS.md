# ControlNet Sketch-to-Image

## 环境
- 推理机：NVIDIA GPU，Python 3.9，CUDA 11.8（AutoDL 云端）
- 本地：Windows A卡，仅运行前端和代码编辑

## 技术栈
- AI推理：diffusers + controlnet_aux
- Backend：FastAPI + Celery + Redis
- Frontend：React + Fabric.js + Vite

## 代码规范
- 注释统一用中文
- 所有配置项从 config.yaml 读取，禁止硬编码
- API统一返回格式：{"code": 0, "data": {}, "msg": "ok"}
- 生成接口必须异步，返回 task_id，前端轮询
- 错误统一用 HTTPException 抛出

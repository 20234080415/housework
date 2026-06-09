param(
    [string]$Python = "C:/Users/Yin/miniconda3/envs/housework/python.exe"
)

$ErrorActionPreference = "Stop"
$backendDir = Resolve-Path "$PSScriptRoot/../backend"
Set-Location $backendDir

# Windows 上 Celery prefork 不稳定，使用 solo 池运行推理 Worker。
& $Python -m celery -A tasks.celery_app worker --loglevel=info --pool=solo

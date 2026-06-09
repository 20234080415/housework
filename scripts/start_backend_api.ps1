param(
    [string]$Python = "C:/Users/Yin/miniconda3/envs/housework/python.exe"
)

$ErrorActionPreference = "Stop"
$backendDir = Resolve-Path "$PSScriptRoot/../backend"
Set-Location $backendDir

& $Python main.py

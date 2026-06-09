param(
    [string]$ModelRoot = "E:/dp_design/housework/models"
)

$ErrorActionPreference = "Stop"

$sdRepo = "runwayml/stable-diffusion-v1-5"
$controlnetRepo = "lllyasviel/sd-controlnet-canny"
$sdDir = Join-Path $ModelRoot "sd-v1-5"
$controlnetDir = Join-Path $ModelRoot "controlnet-canny"

function Test-DirectoryNotEmpty {
    param([string]$Path)
    return (Test-Path $Path) -and ((Get-ChildItem -LiteralPath $Path -Force | Select-Object -First 1) -ne $null)
}

function Download-Model {
    param(
        [string]$RepoId,
        [string]$TargetDir,
        [string]$ModelName
    )

    if (Test-DirectoryNotEmpty $TargetDir) {
        Write-Host "$ModelName 已存在：$TargetDir，跳过下载"
        return
    }

    Write-Host "开始下载 $ModelName 到 $TargetDir"
    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
    huggingface-cli download $RepoId --local-dir $TargetDir --local-dir-use-symlinks False
    Write-Host "$ModelName 下载完成：$TargetDir"
}

New-Item -ItemType Directory -Force -Path $ModelRoot | Out-Null
Write-Host "模型下载目录：$ModelRoot"
Download-Model $sdRepo $sdDir "Stable Diffusion v1.5"
Download-Model $controlnetRepo $controlnetDir "ControlNet Canny"
Write-Host "全部模型检查完成"

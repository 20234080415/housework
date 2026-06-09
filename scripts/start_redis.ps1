param(
    [string]$MemuraiExe = "E:/dp_design/housework/tools/MemuraiDeveloper/tools/memurai.exe",
    [int]$Port = 6379
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $MemuraiExe)) {
    throw "未找到 Memurai：$MemuraiExe，请先安装或解压 Memurai Developer。"
}

# 作为前台进程启动，便于在终端中查看 Redis 兼容服务日志。
& $MemuraiExe --port $Port

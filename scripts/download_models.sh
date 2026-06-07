#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="/root/models"
SD_REPO="runwayml/stable-diffusion-v1-5"
CONTROLNET_REPO="lllyasviel/sd-controlnet-canny"
SD_DIR="${MODEL_ROOT}/sd-v1-5"
CONTROLNET_DIR="${MODEL_ROOT}/controlnet-canny"

# 创建模型根目录。
mkdir -p "${MODEL_ROOT}"

has_command() {
  command -v "$1" >/dev/null 2>&1
}

download_with_modelscope() {
  local repo_id="$1"
  local target_dir="$2"

  echo "使用 ModelScope 下载 ${repo_id} 到 ${target_dir}"
  modelscope download \
    --model "${repo_id}" \
    --local_dir "${target_dir}"
}

download_with_huggingface() {
  local repo_id="$1"
  local target_dir="$2"

  echo "使用 Hugging Face CLI 下载 ${repo_id} 到 ${target_dir}"
  huggingface-cli download "${repo_id}" \
    --local-dir "${target_dir}" \
    --local-dir-use-symlinks False
}

download_model() {
  local repo_id="$1"
  local target_dir="$2"
  local model_name="$3"

  if [ -d "${target_dir}" ] && [ "$(find "${target_dir}" -mindepth 1 -print -quit)" ]; then
    echo "${model_name} 已存在：${target_dir}，跳过下载"
    return
  fi

  echo "开始下载 ${model_name}"
  mkdir -p "${target_dir}"

  if has_command modelscope; then
    download_with_modelscope "${repo_id}" "${target_dir}"
  elif has_command huggingface-cli; then
    download_with_huggingface "${repo_id}" "${target_dir}"
  else
    echo "未找到 modelscope 或 huggingface-cli，请先安装其中一个工具"
    echo "推荐：pip install modelscope"
    exit 1
  fi

  echo "${model_name} 下载完成：${target_dir}"
}

echo "模型下载目录：${MODEL_ROOT}"
download_model "${SD_REPO}" "${SD_DIR}" "Stable Diffusion v1.5"
download_model "${CONTROLNET_REPO}" "${CONTROLNET_DIR}" "ControlNet Canny"
echo "全部模型检查完成"

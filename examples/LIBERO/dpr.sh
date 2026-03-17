#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export DEST=/path/to/dir && bash examples/LIBERO/data_preparation.sh
# or
#   bash examples/LIBERO/data_preparation.sh /path/to/dir

DEST="${DEST:-${1:-}}"
if [[ -z "${DEST}" ]]; then
  echo "ERROR: DEST is not set."
  echo "  export DEST=/path/to/dir && bash examples/LIBERO/data_preparation.sh"
  echo "  or: bash examples/LIBERO/data_preparation.sh /path/to/dir"
  exit 1
fi

CUR="$(pwd)"
mkdir -p "$DEST"

# ---------------------------
# Tunables: 可通过环境变量覆盖
# ---------------------------
: "${HF_DOWNLOAD_RETRIES:=5}"       # 最大重试次数（默认 5 次）
: "${HF_DOWNLOAD_BACKOFF:=5}"       # 回退基数（秒），sleep = backoff * attempt
: "${HF_HUB_ETAG_TIMEOUT:=60}"      # metadata etag 请求超时（秒，默认 HuggingFace 为 10s）
: "${HF_HUB_DOWNLOAD_TIMEOUT:=120}" # 下载超时（秒，尝试性设置）
: "${REQUEST_TIMEOUT:=120}"         # 备用超时 env var（尝试性设置）
# 如果你网络特别差，可以把上面数值调大，例如 HF_HUB_DOWNLOAD_TIMEOUT=300

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_ETAG_TIMEOUT
export HF_HUB_DOWNLOAD_TIMEOUT
export REQUEST_TIMEOUT

# Install huggingface-hub with hf_transfer support (more resilient/faster transfers)
python -m pip install -U "huggingface-hub[hf_transfer]==0.35.3"

# ---------------------------
# Helper: 带重试的 hf download 封装
# ---------------------------
download_with_retries() {
  local repo="$1"
  local repo_type="$2"
  local dest="$3"

  local attempt=0
  local max_attempts="$HF_DOWNLOAD_RETRIES"
  local backoff="$HF_DOWNLOAD_BACKOFF"

  while true; do
    attempt=$((attempt + 1))
    echo ">>> Attempt ${attempt}/${max_attempts}: hf download ${repo} --repo-type ${repo_type} --local-dir ${dest}"
    set +e
    hf download "$repo" --repo-type "$repo_type" --local-dir "$dest"
    rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then
      echo ">>> Success: $repo"
      return 0
    fi

    echo ">>> hf download failed (exit $rc) for $repo"
    if (( attempt >= max_attempts )); then
      echo ">>> ERROR: Reached max attempts ($max_attempts) for $repo"
      return $rc
    fi

    sleep_time=$(( backoff * attempt ))
    echo ">>> Retrying in ${sleep_time}s..."
    sleep "${sleep_time}"
  done
}

# Helper: unzip with retries
unzip_with_retries() {
  local zipfile="$1"
  local destdir="$2"

  local attempt=0
  local max_attempts="$HF_DOWNLOAD_RETRIES"
  local backoff="$HF_DOWNLOAD_BACKOFF"

  while true; do
    attempt=$((attempt + 1))
    echo ">>> Attempt ${attempt}/${max_attempts}: unzip -o -- \"$zipfile\" -d \"$destdir\""
    set +e
    unzip -o -- "$zipfile" -d "$destdir"
    rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then
      echo ">>> unzip succeeded: $zipfile"
      return 0
    fi

    echo ">>> unzip failed (exit $rc) for $zipfile"
    if (( attempt >= max_attempts )); then
      echo ">>> ERROR: unzip reached max attempts ($max_attempts) for $zipfile"
      return $rc
    fi

    sleep_time=$(( backoff * attempt ))
    echo ">>> Retrying unzip in ${sleep_time}s..."
    sleep "${sleep_time}"
  done
}

# ---------------------------
# Downloads (使用重试封装)
# ---------------------------
for repo in \
  IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot
do
  dest_dir="$DEST/libero/${repo##*/}"
  mkdir -p "$dest_dir"
  download_with_retries "$repo" "dataset" "$dest_dir"
done

# LLaVA dataset
mkdir -p "$DEST/LLaVA-OneVision-COCO"
download_with_retries "StarVLA/LLaVA-OneVision-COCO" "dataset" "$DEST/LLaVA-OneVision-COCO"

# unzip the specific zip (如果 zip 已存在且完整，unzip -o 会覆盖)
ZIPFILE="$DEST/LLaVA-OneVision-COCO/sharegpt4v_coco.zip"
if [[ -f "$ZIPFILE" ]]; then
  unzip_with_retries "$ZIPFILE" "$DEST/LLaVA-OneVision-COCO/"
else
  echo ">>> WARNING: expected zipfile not found: $ZIPFILE (skipping unzip)"
fi

# create playground links (保持你原始逻辑)
mkdir -p "$CUR/playground/Datasets"
ln -sfn "$DEST/libero" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA"
ln -sfn "$DEST/LLaVA-OneVision-COCO" "$CUR/playground/Datasets/LLaVA-OneVision-COCO"

## move modality (copy)
mkdir -p "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot/meta"
mkdir -p "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot/meta"
mkdir -p "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot/meta"
mkdir -p "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot/meta"

cp "$CUR/examples/LIBERO/train_files/modality.json" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot/meta"
cp "$CUR/examples/LIBERO/train_files/modality.json" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot/meta"
cp "$CUR/examples/LIBERO/train_files/modality.json" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot/meta"
cp "$CUR/examples/LIBERO/train_files/modality.json" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot/meta"

echo ">>> All done."
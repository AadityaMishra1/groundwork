#!/usr/bin/env bash
# Bootstrap a fresh RunPod GPU pod for Isaac Lab training.
#
# Tested target: RunPod "RunPod Pytorch 2.x" or plain Ubuntu 22.04 + CUDA 12.x
# template, RTX 4090 / L40S / A40, >=50 GB volume mounted at /workspace.
#
# Usage (on the pod):  bash setup_pod.sh
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
VENV="$WORKSPACE/venv"
ISAACLAB_VERSION="${ISAACLAB_VERSION:-2.3.0}"

echo "=== [1/5] System deps ==="
apt-get update -qq
apt-get install -y -qq git git-lfs cmake build-essential ffmpeg \
    libglib2.0-0 libxrandr2 libxinerama1 libxcursor1 libxi6 vulkan-tools

echo "=== [2/5] GPU check ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || {
    echo "FATAL: no NVIDIA GPU visible"; exit 1; }

echo "=== [3/5] Python env (3.11) ==="
if ! command -v python3.11 >/dev/null; then
    apt-get install -y -qq software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq && apt-get install -y -qq python3.11 python3.11-venv
fi
python3.11 -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip

echo "=== [4/5] Isaac Sim + Isaac Lab (pip install, headless) ==="
# Pinned; includes Isaac Sim runtime, rsl_rl, rl_games, and all extensions.
pip install "isaaclab[isaacsim,all]==${ISAACLAB_VERSION}" \
    --extra-index-url https://pypi.nvidia.com

# Accept EULA non-interactively for headless first launch
export OMNI_KIT_ACCEPT_EULA=YES

echo "=== [5/5] Smoke test ==="
cd "$(dirname "$0")"
python smoke_test.py --headless && echo "POD READY" || {
    echo "Smoke test failed — see output above."; exit 1; }

echo
echo "Add to ~/.bashrc:  source $VENV/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES"

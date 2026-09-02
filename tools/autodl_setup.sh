#!/bin/bash
# AutoDL 部署脚本：lewm 环境 + Push-T baseline 训练
# 使用方法：上传到 AutoDL 终端后执行 bash autodl_setup.sh

set -e

echo "=== 步骤 1: 检查 AutoDL 环境 ==="

# 检查数据盘挂载点
if [ -d "/root/autodl-tmp" ]; then
    DATA_DIR="/root/autodl-tmp"
elif [ -d "/root/autodl-fs" ]; then
    DATA_DIR="/root/autodl-fs"
else
    DATA_DIR="/root"
fi

export STABLEWM_HOME="$DATA_DIR/.stable_worldmodel"
mkdir -p "$STABLEWM_HOME"
echo "数据目录设置为: $STABLEWM_HOME"

# 检查 conda
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 conda。请在 AutoDL 选择带有 conda 的镜像。"
    exit 1
fi

echo "=== 步骤 2: 创建 conda 环境 ==="
conda create -n lewm python=3.12 -y || echo "环境已存在，跳过"

# 激活环境（兼容脚本）
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lewm

echo "=== 步骤 3: 检查 PyTorch / CUDA ==="
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('gpu:', torch.cuda.get_device_name(0))" || {
    echo "PyTorch 未安装或版本不对，开始安装..."
    pip install torch==2.12.1 torchvision --index-url https://download.pytorch.org/whl/cu130
}

echo "=== 步骤 4: 安装 stable-worldmodel [train] ==="
# 假设脚本在仓库根目录执行
pip install -e ".[train]" --no-cache-dir

echo "=== 步骤 5: 下载 Push-T 数据集 ==="
# 国内用户如果 HuggingFace 慢，取消下面这行注释
# export HF_ENDPOINT=https://hf-mirror.com
python download_pusht.py

echo "=== 步骤 6: 启动 LeWM baseline 训练 ==="
python scripts/train/lewm.py data=pusht_hf

echo "=== 部署完成 ==="

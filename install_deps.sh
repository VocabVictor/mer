#!/bin/bash

ENV_NAME=${ENV_NAME:-llm}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
CUDA_VERSION=${CUDA_VERSION:-11.8}

echo "安装 Mamba 项目依赖 (CUDA $CUDA_VERSION)..."

source ~/.bashrc

# 初始化 micromamba
if ! command -v micromamba &> /dev/null; then
    if [ -n "$MAMBA_EXE" ] && [ -f "$MAMBA_EXE" ]; then
        eval "$("$MAMBA_EXE" shell hook --shell bash 2>/dev/null)" 2>/dev/null || alias micromamba="$MAMBA_EXE"
    fi
fi

if ! command -v micromamba &> /dev/null; then
    echo "错误: 找不到 micromamba"
    return 1 2>/dev/null || exit 1
fi

# 创建并激活环境
micromamba env list | grep -q "$ENV_NAME" || micromamba create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
micromamba activate "$ENV_NAME"

# 安装系统级依赖（仅在未安装时执行）
MISSING_PKGS=""
for pkg in ninja cmake gcc_linux-64 gxx_linux-64; do
    micromamba list | grep -q "^$pkg " || MISSING_PKGS="$MISSING_PKGS $pkg"
done

# 检查CUDA包
for pkg in cuda-toolkit cuda-nvcc cuda-cudart cuda-cudart-dev cuda-nvrtc cuda-nvrtc-dev; do
    micromamba list | grep -q "^$pkg " || MISSING_PKGS="$MISSING_PKGS $pkg=$CUDA_VERSION"
done

if [ -n "$MISSING_PKGS" ]; then
    micromamba install -y $MISSING_PKGS -c "nvidia/label/cuda-$CUDA_VERSION.0" -c nvidia
fi

export PATH="$CONDA_PREFIX/bin:$PATH"

# 设置CUDA环境变量
if command -v nvcc &> /dev/null; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export FORCE_CUDA=1
fi

# 安装Python依赖
CUDA_SHORT=$(echo "$CUDA_VERSION" | tr -d '.')
# 先安装uv
command -v uv &> /dev/null || pip install uv

# 检查CUDA_SHORT并安装对应版本的PyTorch
echo "CUDA版本: $CUDA_VERSION, CUDA_SHORT: $CUDA_SHORT"
if [ "$CUDA_SHORT" = "118" ]; then
    # CUDA 11.8 版本
    echo "安装 PyTorch CUDA 11.8 版本..."
    uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url "https://download.pytorch.org/whl/cu118"
elif [ "$CUDA_SHORT" = "126" ]; then
    # CUDA 12.6 版本 
    echo "安装 PyTorch CUDA 12.6 版本..."
    uv pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cu126"
else
    echo "警告: 不支持的CUDA版本 $CUDA_VERSION，尝试安装通用版本..."
    uv pip install torch torchvision torchaudio
fi

uv pip install -r requirements.txt

echo "安装完成! 验证: micromamba activate $ENV_NAME && python mamba_example.py" 
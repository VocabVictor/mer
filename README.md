# mer

## 安装依赖

由于以下原因，直接使用 `pip install -r requirements.txt` 会失败：
1. `causal-conv1d` 包在构建时需要 `torch` 但没有正确声明构建依赖
2. 需要 CUDA 版本与 PyTorch 版本兼容（原系统 CUDA 11.8 与新版 PyTorch 不兼容）

因此我们需要先安装 CUDA 12.1，然后按正确顺序安装依赖。

### 推荐安装方法

#### 选项1: 自动安装脚本（推荐）
```bash
bash install_deps.sh
```
该脚本会自动检测 micromamba 安装位置，适用于大多数环境。

#### 选项2: 简化安装脚本（通用性更好）
如果选项1无法识别您的 micromamba 环境，请手动激活环境后使用简化脚本：
```bash
micromamba activate base
bash install_deps_simple.sh
```

#### 选项3: 不含CUDA的安装脚本（最通用）
如果您想要手动处理CUDA兼容性，或者环境检测有问题：
```bash
bash install_deps_no_cuda.sh
```

#### 选项4: 完全手动安装（最可靠）
如果所有脚本都有问题，请按照下面的手动安装步骤进行。

### 手动安装方法

如果需要手动安装，请按以下顺序：

1. 先安装 CUDA 12.1 到 base 环境：
```bash
micromamba install -y cuda-toolkit=12.1 -c nvidia
```

2. 安装 torch 和基础依赖：
```bash
uv pip install torch>=2.0.0 numpy einops transformers>=4.30.0
```

3. 使用 --no-build-isolation 安装需要 torch 的包：
```bash
uv pip install causal-conv1d>=1.4.0 --no-build-isolation
uv pip install mamba-ssm --no-build-isolation
```

4. 安装可选依赖：
```bash
uv pip install jupyter matplotlib tqdm
```

## 验证安装

安装完成后，可以运行以下命令验证：
```bash
python -c "import torch; import mamba_ssm; print('安装成功！')"
```

## 故障排除

### 问题1: 脚本中找不到 micromamba 或 uv
**原因**: Shell脚本不会自动加载交互式shell的配置（~/.bashrc）

**解决方案**:
1. 手动激活环境：`micromamba activate base`
2. 然后使用手动安装方法，或者
3. 在当前shell中直接运行安装命令

### 问题2: CUDA版本不匹配
**原因**: 系统CUDA版本与PyTorch编译版本不匹配

**解决方案**:
1. 检查CUDA版本: `nvcc --version`
2. 如果是CUDA 11.8，安装对应版本: `uv pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118`
3. 或者升级CUDA到12.1+

### 问题3: causal-conv1d 构建失败
**原因**: 构建时找不到torch

**解决方案**: 使用 `--no-build-isolation` 参数：
```bash
uv pip install causal-conv1d>=1.4.0 --no-build-isolation
```
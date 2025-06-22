#!/usr/bin/env python3
"""
Mamba 最简单的使用示例
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def run_simple_demo(device):
    """运行简化版 Mamba 概念演示"""
    print("\n📝 Mamba概念演示 (简化版):")
    
    class SimpleMamba(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.d_model = d_model
            self.linear1 = nn.Linear(d_model, d_model * 2)
            self.conv1d = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model)
            self.linear2 = nn.Linear(d_model, d_model)
            
        def forward(self, x):
            # x: (batch, seq_len, d_model)
            residual = x
            x = self.linear1(x)  # 扩展
            x1, x2 = x.chunk(2, dim=-1)  # 分割
            
            # 简化的SSM操作 (这里用卷积来模拟)
            x1_conv = self.conv1d(x1.transpose(1, 2)).transpose(1, 2)
            x1 = F.silu(x1_conv)  # 使用 F.silu 而不是 torch.silu
            
            # 门控机制
            x = x1 * torch.sigmoid(x2)
            
            # 输出投影
            x = self.linear2(x)
            return x + residual
    
    # 测试简化版本
    model = SimpleMamba(d_model=256).to(device)
    x = torch.randn(2, 64, 256).to(device)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"简化版输入形状: {x.shape}")
    print(f"简化版输出形状: {output.shape}")
    print("✅ 概念演示完成！")

def main():
    print("🐍 Mamba 最简单使用示例")
    print("=" * 50)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    print("\n📦 安装说明:")
    print("pip install causal-conv1d>=1.4.0")
    print("pip install mamba-ssm")
    print("或者: pip install 'mamba-ssm[causal-conv1d]'")
    
    print("\n🔧 示例1: 基本 Mamba 块")
    try:
        from mamba_ssm import Mamba
        
        # 配置参数
        batch_size = 2
        seq_length = 64
        d_model = 256
        
        # 创建输入数据
        x = torch.randn(batch_size, seq_length, d_model).to(device)
        print(f"输入形状: {x.shape}")
        
        # 创建 Mamba 模型
        model = Mamba(
            d_model=d_model,    # 模型维度
            d_state=16,         # SSM 状态扩展因子
            d_conv=4,           # 本地卷积宽度
            expand=2,           # 块扩展因子
        ).to(device)
        
        # 前向传播
        with torch.no_grad():
            output = model(x)
        
        print(f"输出形状: {output.shape}")
        print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print("✅ 基本 Mamba 块运行成功！")
        
    except ImportError as e:
        print("❌ 未安装 mamba-ssm，请先安装:")
        print("pip install mamba-ssm")
        print(f"错误详情: {e}")
        
        # 运行简化版本来演示概念
        run_simple_demo(device)
    
    except Exception as e:
        print(f"❌ mamba-ssm 运行出错: {e}")
        print("这可能是由于 CUDA 版本不兼容导致的")
        print("建议尝试重新编译安装 mamba-ssm:")
        print("pip uninstall mamba-ssm causal-conv1d")
        print("pip install causal-conv1d>=1.4.0 --no-cache-dir")
        print("pip install mamba-ssm --no-cache-dir")
        
        # 运行简化版本
        run_simple_demo(device)
    
    print("\n🚀 示例2: 预训练模型使用框架")
    print("# 加载预训练模型的代码示例:")
    print("from transformers import AutoTokenizer")
    print("from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel")
    print("")
    print("# 加载模型和分词器")
    print("tokenizer = AutoTokenizer.from_pretrained('state-spaces/mamba-130m')")
    print("model = MambaLMHeadModel.from_pretrained('state-spaces/mamba-130m')")
    print("")
    print("# 推理")
    print("text = 'Hello, how are you?'")
    print("tokens = tokenizer.encode(text, return_tensors='pt')")
    print("with torch.no_grad():")
    print("    output = model(tokens)")
    print("    logits = output.logits")
    
    print("\n📚 学习资源:")
    print("- 官方仓库: https://github.com/state-spaces/mamba")
    print("- 论文: https://arxiv.org/abs/2312.00752")
    print("- HuggingFace模型: https://huggingface.co/state-spaces")
    
    print("\n🎉 示例完成！")

if __name__ == "__main__":
    main() 
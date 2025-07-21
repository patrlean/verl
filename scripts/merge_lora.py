#!/usr/bin/env python3
"""
合并LoRA权重到基础模型的脚本
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora_model():
    # 配置路径
    # base_model_path = "Qwen/Qwen3-8B-Base"  # 或使用本地路径
    base_model_path = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
    
    lora_adapter_path = "/workspace/qwen3-8b-base-grpo-w-sft/qwen3-8b-base_gsm8k_simplerl_grpo_lora_8192/global_step_40/actor/lora_adapter"
    output_path = "/workspace/merged_qwen3_grpo_model"
    
    print("开始合并LoRA模型...")
    print(f"基础模型路径: {base_model_path}")
    print(f"LoRA适配器路径: {lora_adapter_path}")
    print(f"输出路径: {output_path}")
    
    # 首先修复adapter_config.json中的base_model_name_or_path
    adapter_config_path = os.path.join(lora_adapter_path, "adapter_config.json")
    print(f"\n修复adapter配置文件: {adapter_config_path}")
    
    with open(adapter_config_path, "r") as f:
        adapter_config = json.load(f)
    
    # 更新base_model_name_or_path为正确的路径
    adapter_config["base_model_name_or_path"] = base_model_path
    
    with open(adapter_config_path, "w") as f:
        json.dump(adapter_config, f, indent=4)
    
    print("✓ 已修复adapter配置文件")
    
    # 加载基础模型
    print(f"\n加载基础模型: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("✓ 基础模型加载完成")
    
    # 加载tokenizer
    print(f"加载tokenizer: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    print("✓ Tokenizer加载完成")
    
    # 加载LoRA适配器
    print(f"\n加载LoRA适配器: {lora_adapter_path}")
    model_with_lora = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        torch_dtype=torch.bfloat16
    )
    print("✓ LoRA适配器加载完成")
    
    # 合并权重
    print("\n合并LoRA权重到基础模型...")
    merged_model = model_with_lora.merge_and_unload()
    print("✓ 权重合并完成")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 保存合并后的模型
    print(f"\n保存合并后的模型到: {output_path}")
    merged_model.save_pretrained(
        output_path,
        torch_dtype=torch.bfloat16,
        safe_serialization=True
    )
    print("✓ 模型保存完成")
    
    # 保存tokenizer
    print("保存tokenizer...")
    tokenizer.save_pretrained(output_path)
    print("✓ Tokenizer保存完成")
    
    print(f"\n🎉 合并完成！合并后的模型保存在: {output_path}")
    print(f"现在您可以使用以下路径进行推理:")
    print(f"model.path={output_path}")
    
    return output_path

if __name__ == "__main__":
    try:
        merged_path = merge_lora_model()
        
    except Exception as e:
        print(f"❌ 合并过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
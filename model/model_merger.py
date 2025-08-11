# model/model_merger.py
"""
模型合并工具
"""
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class ModelMerger:
    """模型合并器"""
    
    @staticmethod
    def merge_lora_model(base_model_path, lora_adapter_path, merged_model_path):
        """
        合并LoRA模型到基础模型中
        
        Args:
            base_model_path (str): 基础模型路径
            lora_adapter_path (str): LoRA适配器路径
            merged_model_path (str): 合并后模型保存路径
        """
        print("=== LoRA模型合并工具 ===")
        print(f"基础模型路径: {base_model_path}")
        print(f"LoRA适配器路径: {lora_adapter_path}")
        print(f"合并后模型路径: {merged_model_path}")
        
        # 创建输出目录
        os.makedirs(merged_model_path, exist_ok=True)
        
        try:
            # 加载基础模型
            print("1/4 加载基础模型...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            print("基础模型加载完成")
            
            # 加载LoRA适配器
            print("2/4 加载LoRA适配器...")
            peft_model = PeftModel.from_pretrained(
                base_model,
                lora_adapter_path,
                torch_dtype=torch.float16
            )
            print("LoRA适配器加载完成")
            
            # 合并权重
            print("3/4 合并权重...")
            merged_model = peft_model.merge_and_unload()
            print("权重合并完成")
            
            # 加载分词器
            print("4/4 加载并保存分词器...")
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True
            )
            
            # 保存合并后的模型
            print("保存合并后的模型...")
            merged_model.save_pretrained(
                merged_model_path, 
                safe_serialization=True,
                max_shard_size="2GB"  # 控制分片大小
            )
            tokenizer.save_pretrained(merged_model_path)
            
            print("=== 合并完成 ===")
            print(f"合并后的模型已保存到: {merged_model_path}")
            
            # 显示模型信息
            total_params = sum(p.numel() for p in merged_model.parameters())
            print(f"模型总参数量: {total_params:,}")
            
        except Exception as e:
            print(f"合并过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
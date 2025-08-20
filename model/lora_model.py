# model/lora_model.py
"""
LoRA模型创建和配置
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import torch

def create_lora_model(model_path: str, config=None):
    """
    创建LoRA模型
    
    Args:
        model_path: 模型路径
        config: 训练配置对象，如果为None则使用默认配置
    """
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='right'
    )
    
    # 如果没有pad token则添加
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation=None,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # 设置pad token id
    if hasattr(model.config, 'pad_token_id') and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # 配置LoRA参数
    lora_r = config.LORA_R if config else 8
    lora_alpha = config.LORA_ALPHA if config else 16
    lora_dropout = config.LORA_DROPOUT if config else 0.1
    target_modules = config.LORA_TARGET_MODULES if config else ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # 应用LoRA到模型
    model = get_peft_model(model, lora_config)
    
    # 如果模型仍在meta设备上，将其移动到正确的设备
    if any(param.is_meta for param in model.parameters()):
        model = model.to_empty(device="cpu" if not torch.cuda.is_available() else "cuda:0")
    
    # 确保模型在正确的设备上
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    return model, tokenizer
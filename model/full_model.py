# model/full_model.py
"""
全量微调模型创建和配置
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def create_full_model(model_path: str, config=None):
    """
    创建全量微调模型
    
    Args:
        model_path: 模型路径
        config: 训练配置对象
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
    
    # 加载模型（全量微调，不使用LoRA）
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # 设置pad token id
    if hasattr(model.config, 'pad_token_id') and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # 如果模型仍在meta设备上，将其移动到正确的设备
    if any(param.is_meta for param in model.parameters()):
        model = model.to_empty(device="cpu" if not torch.cuda.is_available() else "cuda:0")
    
    # 确保模型在正确的设备上
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    return model, tokenizer

def load_full_model(model_name, **kwargs):
    """
    加载完整模型用于全量微调
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model

def freeze_model_layers(model, freeze_layers=False):
    """
    可选地冻结部分层
    """
    if freeze_layers:
        for name, param in model.named_parameters():
            if 'embed' in name or 'norm' in name:
                param.requires_grad = False
    return model
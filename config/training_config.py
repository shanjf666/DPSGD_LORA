# config/training_config.py
"""
训练配置参数
"""

class TrainingConfig:
    # 模型路径
    MODEL_PATH = "/root/autodl-tmp/final/model8"
    
    # 数据路径
    TRAIN_DATA_PATH = "/root/autodl-tmp/data/privacy_train.jsonl"
    VALID_DATA_PATH = "/root/autodl-tmp/data/valid.jsonl"
    OUTPUT_DIR = "/root/autodl-tmp/output/output9"
    
    # 模型合并路径配置
    MERGED_MODEL_PATH = "/root/autodl-tmp/final/model9"
    
    # 训练参数（优化内存使用）
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    PER_DEVICE_EVAL_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 16
    NUM_TRAIN_EPOCHS = 2
    LEARNING_RATE = 2e-4
    MAX_LENGTH = 256
    
    # LoRA微调参数
    LORA_R = 8
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    # 传统差分隐私参数
    TARGET_EPSILON = 1.0
    TARGET_DELTA = 1e-6
    MAX_GRAD_NORM = 1.0
    
    # 其他参数
    LOGGING_STEPS = 100
    SAVE_STEPS = 100
    EVAL_STEPS = 5000
    SAVE_TOTAL_LIMIT = 2
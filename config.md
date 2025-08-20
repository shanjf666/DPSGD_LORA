# 训练配置参数

总共训练了8次，每⼀次训练是在前⼀次训练好的模型上训练的，第⼀次训练使⽤的是基础模型。

以下是训练配置参数：

## train1

```python
class TrainingConfig:
    # 模型路径 
    MODEL_PATH = "/root/autodl-tmp/model"
    
    # 数据路径 
    TRAIN_DATA_PATH = "/root/autodl-tmp/data/privacy_train.jsonl"
    VALID_DATA_PATH = "/root/autodl-tmp/data/valid.jsonl"
    OUTPUT_DIR = "/root/autodl-tmp/output/output1"
    
    # 模型合并路径配置 
    MERGED_MODEL_PATH = "/root/autodl-tmp/final/model1"
    
    # 训练参数（优化内存使⽤） 
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    PER_DEVICE_EVAL_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 16
    NUM_TRAIN_EPOCHS = 2
    LEARNING_RATE = 2e-4
    MAX_LENGTH = 256
    
    # LoRA微调参数 
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    # 传统差分隐私参数 
    TARGET_EPSILON = 3.0
    TARGET_DELTA = 2e-6
    MAX_GRAD_NORM = 1.0
    
    # 其他参数 
    LOGGING_STEPS = 100
    SAVE_STEPS = 100
    EVAL_STEPS = 5000
    SAVE_TOTAL_LIMIT = 2
```

## train2

```python
class TrainingConfig:
    # 模型路径
    MODEL_PATH = "/root/autodl-tmp/final/model1"
    
    # 数据路径
    TRAIN_DATA_PATH = "/root/autodl-tmp/data/privacy_train.jsonl"
    VALID_DATA_PATH = "/root/autodl-tmp/data/valid.jsonl"
    OUTPUT_DIR = "/root/autodl-tmp/output/output2"
    
    # 模型合并路径配置
    MERGED_MODEL_PATH = "/root/autodl-tmp/final/model2"
    
    # 训练参数（优化内存使用）
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    PER_DEVICE_EVAL_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 16
    NUM_TRAIN_EPOCHS = 2
    LEARNING_RATE = 2e-4
    MAX_LENGTH = 256
    
    # LoRA微调参数
    LORA_R = 8
    LORA_ALPHA = 16
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
```

## train3

```python
class TrainingConfig:
    # 模型路径
    MODEL_PATH = "/root/autodl-tmp/final/model2"
    
    # 数据路径
    TRAIN_DATA_PATH = "/root/autodl-tmp/data/privacy_train.jsonl"
    VALID_DATA_PATH = "/root/autodl-tmp/data/valid.jsonl"
    OUTPUT_DIR = "/root/autodl-tmp/output/output3"
    
    # 模型合并路径配置
    MERGED_MODEL_PATH = "/root/autodl-tmp/final/model3"
    
    # 训练参数（优化内存使用）
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    PER_DEVICE_EVAL_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 16
    NUM_TRAIN_EPOCHS = 2
    LEARNING_RATE = 2e-4
    MAX_LENGTH = 256
    
    # LoRA微调参数
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    # 传统差分隐私参数
    TARGET_EPSILON = 0.1
    TARGET_DELTA = 1e-7
    MAX_GRAD_NORM = 0.1
    
    # 其他参数
    LOGGING_STEPS = 100
    SAVE_STEPS = 100
    EVAL_STEPS = 5000
    SAVE_TOTAL_LIMIT = 2
```

## train4

```python
class TrainingConfig:
    # 模型路径
    MODEL_PATH = "/root/autodl-tmp/final/model3"
    
    # 数据路径
    TRAIN_DATA_PATH = "/root/autodl-tmp/data/privacy_train.jsonl"
    VALID_DATA_PATH = "/root/autodl-tmp/data/valid.jsonl"
    OUTPUT_DIR = "/root/autodl-tmp/output/output4"
    
    # 模型合并路径配置
    MERGED_MODEL_PATH = "/root/autodl-tmp/final/model4"
    
    # 训练参数（优化内存使用）
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    PER_DEVICE_EVAL_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 16
    NUM_TRAIN_EPOCHS = 2
    LEARNING_RATE = 1e-4
    MAX_LENGTH = 256
    
    # LoRA微调参数
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    # 传统差分隐私参数
    TARGET_EPSILON = 0.01
    TARGET_DELTA = 1e-7
    MAX_GRAD_NORM = 0.1
    
    # 其他参数
    LOGGING_STEPS = 100
    SAVE_STEPS = 100
    EVAL_STEPS = 5000
    SAVE_TOTAL_LIMIT = 2
```

## train5

```python
class TrainingConfig:
    # 模型路径
    MODEL_PATH = "/root/autodl-tmp/final/model4"
    
    # 数据路径
    TRAIN_DATA_PATH = "/root/autodl-tmp/data/privacy_train.jsonl"
    VALID_DATA_PATH = "/root/autodl-tmp/data/valid.jsonl"
    OUTPUT_DIR = "/root/autodl-tmp/output/output5"
    
    # 模型合并路径配置
    MERGED_MODEL_PATH = "/root/autodl-tmp/final/model5"
    
    # 训练参数（优化内存使用）
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    PER_DEVICE_EVAL_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 16
    NUM_TRAIN_EPOCHS = 2
    LEARNING_RATE = 1e-4
    MAX_LENGTH = 256
    
    # LoRA微调参数
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    # 传统差分隐私参数
    TARGET_EPSILON = 0.01
    TARGET_DELTA = 1e-7
    MAX_GRAD_NORM = 0.01
    
    # 其他参数
    LOGGING_STEPS = 100
    SAVE_STEPS = 100
    EVAL_STEPS = 5000
    SAVE_TOTAL_LIMIT = 2
```

## train6

```python
class TrainingConfig:
    # 模型路径
    MODEL_PATH = "/root/autodl-tmp/final/model5"
    
    # 数据路径
    TRAIN_DATA_PATH = "/root/autodl-tmp/data/privacy_train.jsonl"
    VALID_DATA_PATH = "/root/autodl-tmp/data/valid.jsonl"
    OUTPUT_DIR = "/root/autodl-tmp/output/output6"
    
    # 模型合并路径配置
    MERGED_MODEL_PATH = "/root/autodl-tmp/final/model6"
    
    # 训练参数（优化内存使用）
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    PER_DEVICE_EVAL_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 16
    NUM_TRAIN_EPOCHS = 3
    LEARNING_RATE = 1e-4
    MAX_LENGTH = 256
    
    # LoRA微调参数
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    # 传统差分隐私参数
    TARGET_EPSILON = 0.001
    TARGET_DELTA = 1e-7
    MAX_GRAD_NORM = 0.001
    
    # 其他参数
    LOGGING_STEPS = 100
    SAVE_STEPS = 100
    EVAL_STEPS = 5000
    SAVE_TOTAL_LIMIT = 2
```

## train7

```python
class TrainingConfig:
    # 模型路径
    MODEL_PATH = "/root/autodl-tmp/final/model6"
    
    # 数据路径
    TRAIN_DATA_PATH = "/root/autodl-tmp/data/privacy_train.jsonl"
    VALID_DATA_PATH = "/root/autodl-tmp/data/valid.jsonl"
    OUTPUT_DIR = "/root/autodl-tmp/output/output7"
    
    # 模型合并路径配置
    MERGED_MODEL_PATH = "/root/autodl-tmp/final/model7"
    
    # 训练参数（优化内存使用）
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    PER_DEVICE_EVAL_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 16
    NUM_TRAIN_EPOCHS = 3
    LEARNING_RATE = 1e-5
    MAX_LENGTH = 256
    
    # LoRA微调参数
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    # 传统差分隐私参数
    TARGET_EPSILON = 0.0001
    TARGET_DELTA = 1e-7
    MAX_GRAD_NORM = 0.0001
    
    # 其他参数
    LOGGING_STEPS = 100
    SAVE_STEPS = 100
    EVAL_STEPS = 5000
    SAVE_TOTAL_LIMIT = 2
```

## train8

```python
class TrainingConfig:
    # 模型路径
    MODEL_PATH = "/root/autodl-tmp/final/model7"
    
    # 数据路径
    TRAIN_DATA_PATH = "/root/autodl-tmp/data/privacy_train.jsonl"
    VALID_DATA_PATH = "/root/autodl-tmp/data/valid.jsonl"
    OUTPUT_DIR = "/root/autodl-tmp/output/output8"
    
    # 模型合并路径配置
    MERGED_MODEL_PATH = "/root/autodl-tmp/final/model8"
    
    # 训练参数（优化内存使用）
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    PER_DEVICE_EVAL_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 16
    NUM_TRAIN_EPOCHS = 3
    LEARNING_RATE = 1e-5
    MAX_LENGTH = 256
    
    # LoRA微调参数
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    # 传统差分隐私参数
    TARGET_EPSILON = 0.0001
    TARGET_DELTA = 1e-8
    MAX_GRAD_NORM = 0.0001
    
    # 其他参数
    LOGGING_STEPS = 100
    SAVE_STEPS = 100
    EVAL_STEPS = 5000
    SAVE_TOTAL_LIMIT = 2
```
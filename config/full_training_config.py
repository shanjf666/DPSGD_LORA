# config/full_training_config.py
"""
全量微调训练配置参数
"""
class FullTrainingConfig:
    def __init__(self):
        # 基础配置
        self.model_name = "qwen/Qwen2.5-1.5B-Instruct"
        self.output_dir = "./full_finetuned_model"
        
        # 数据配置
        self.train_data_path = "./data/privacy_train.jsonl"
        self.eval_data_path = "./data/valid.jsonl"
        self.max_length = 256
        
        # 训练参数
        self.batch_size = 4
        self.per_device_train_batch_size = 1
        self.per_device_eval_batch_size = 1
        self.gradient_accumulation_steps = 16
        self.num_epochs = 3
        self.num_train_epochs = 3
        self.learning_rate = 5e-6  # 全量微调通常需要更小的学习率
        self.weight_decay = 0.01
        self.logging_steps = 10
        self.save_steps = 100
        self.eval_steps = 500
        self.save_total_limit = 2
        
        # 差分隐私配置
        self.use_dp = True
        self.target_epsilon = 1.0
        self.target_delta = 1e-5
        self.max_grad_norm = 1.0
        self.noise_multiplier = 1.0
        
        # 其他参数
        self.LOGGING_STEPS = 10
        self.SAVE_STEPS = 100
        self.EVAL_STEPS = 500
        self.SAVE_TOTAL_LIMIT = 2
        self.OUTPUT_DIR = self.output_dir
        self.PER_DEVICE_TRAIN_BATCH_SIZE = self.per_device_train_batch_size
        self.PER_DEVICE_EVAL_BATCH_SIZE = self.per_device_eval_batch_size
        self.GRADIENT_ACCUMULATION_STEPS = self.gradient_accumulation_steps
        self.NUM_TRAIN_EPOCHS = self.num_train_epochs
        self.LEARNING_RATE = self.learning_rate
        self.MAX_GRAD_NORM = self.max_grad_norm
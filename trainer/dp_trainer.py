# trainer/dp_trainer.py
"""
差分隐私训练器
"""
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
from torch.utils.data import DataLoader

class DPTrainer:
    """差分隐私训练器"""
    
    def __init__(self, config):
        self.config = config
        self.trainer = None
        self.privacy_engine = None
    
    def setup_training(self, model, train_dataset, val_dataset, tokenizer):
        """
        设置训练环境
        
        Args:
            model: 模型
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            tokenizer: 分词器
        """
        # 训练参数设置
        training_args = TrainingArguments(
            output_dir=self.config.OUTPUT_DIR,
            per_device_train_batch_size=self.config.PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=self.config.PER_DEVICE_EVAL_BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION_STEPS,
            num_train_epochs=self.config.NUM_TRAIN_EPOCHS,
            learning_rate=self.config.LEARNING_RATE,
            fp16=False,  # 禁用FP16以确保DP兼容性
            logging_steps=self.config.LOGGING_STEPS,
            save_steps=self.config.SAVE_STEPS,
            eval_steps=self.config.EVAL_STEPS,
            eval_strategy="steps" if hasattr(TrainingArguments, 'eval_strategy') else "steps",
            save_total_limit=self.config.SAVE_TOTAL_LIMIT,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to="none",  # 禁用wandb日志记录
        )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # 初始化训练器
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        return training_args
    
    def apply_manual_privacy(self, model, optimizer, train_dataset, noise_multiplier):
        """
        应用差分隐私
        
        Args:
            model: 模型
            optimizer: 优化器
            train_dataset: 训练数据集
            noise_multiplier: 噪声乘数
        """
        from privacy.manual_dp import make_private
        
        # 创建数据加载器
        data_loader = DataLoader(
            train_dataset,
            batch_size=self.config.PER_DEVICE_TRAIN_BATCH_SIZE,
            shuffle=True
        )
        
        # 应用隐私保护
        private_model, private_optimizer, private_data_loader, privacy_engine = make_private(
            model, optimizer, data_loader, noise_multiplier, self.config.MAX_GRAD_NORM
        )
        
        self.privacy_engine = privacy_engine
        return private_model, private_optimizer, private_data_loader
    
    def train(self, model, train_dataset, val_dataset, tokenizer, noise_multiplier):
        """
        执行训练
        
        Args:
            model: 模型
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            tokenizer: 分词器
            noise_multiplier: 噪声乘数
        """
        # 设置训练
        training_args = self.setup_training(model, train_dataset, val_dataset, tokenizer)
        
        # 应用手动隐私引擎
        if hasattr(self.trainer, 'optimizer'):
            private_model, private_optimizer, private_data_loader = self.apply_manual_privacy(
                model, self.trainer.optimizer, train_dataset, noise_multiplier
            )
            # 更新训练器组件
            self.trainer.model = private_model
            self.trainer.optimizer = private_optimizer
        
        # 开始训练
        self.trainer.train()
        
        return self.trainer
    
    def get_epsilon_spent(self, delta):
        """
        获取消耗的epsilon
        
        Args:
            delta: delta值
            
        Returns:
            epsilon: 消耗的epsilon
        """
        if self.privacy_engine:
            return self.privacy_engine.get_epsilon(delta)
        return 0.0
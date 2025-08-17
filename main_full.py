# main_full.py (补充完整)
import os
import sys
from config.full_training_config import FullTrainingConfig
from model.full_model import create_full_model
from data.medical_dataset import MedicalDataset
from trainer.dp_trainer import DPTrainer
from privacy.dp_calculator import DPNoiseCalculator
from privacy.manual_dp import make_private
from utils.logger import get_logger
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

logger = get_logger(__name__)

def main():
    # 加载配置
    config = FullTrainingConfig()
    
    logger.info("开始全量微调DPSGD训练...")
    logger.info(f"模型路径: {config.model_name}")
    logger.info(f"训练数据路径: {config.train_data_path}")
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 加载模型和分词器
    logger.info("加载模型和分词器...")
    model, tokenizer = create_full_model(config.model_name)
    logger.info(f"模型加载完成: {config.model_name}")
    
    # 加载数据集
    logger.info("加载训练数据集...")
    train_dataset = MedicalDataset(config.train_data_path, tokenizer, config.max_length)
    
    eval_dataset = None
    if config.eval_data_path and os.path.exists(config.eval_data_path):
        logger.info("加载验证数据集...")
        eval_dataset = MedicalDataset(config.eval_data_path, tokenizer, config.max_length)
    
    logger.info(f"训练数据集大小: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"验证数据集大小: {len(eval_dataset)}")
    
    # 初始化训练器
    trainer_wrapper = DPTrainer(config)
    
    # 设置训练参数
    training_args = trainer_wrapper.setup_training(model, train_dataset, eval_dataset, tokenizer)
    
    # 计算噪声乘数
    noise_calculator = DPNoiseCalculator()
    noise_multiplier = noise_calculator.compute_noise_multiplier(
        config.target_epsilon,
        config.target_delta,
        config.max_grad_norm
    )
    logger.info(f"计算得到的噪声乘数: {noise_multiplier}")
    
    # 应用差分隐私
    if config.use_dp:
        logger.info("应用差分隐私保护...")
        data_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        )
        
        private_model, private_optimizer, private_data_loader, privacy_engine = make_private(
            model, trainer_wrapper.trainer.optimizer, data_loader, noise_multiplier, config.max_grad_norm
        )
        
        # 更新训练器组件
        trainer_wrapper.trainer.model = private_model
        trainer_wrapper.trainer.optimizer = private_optimizer
        trainer_wrapper.privacy_engine = privacy_engine
        logger.info("差分隐私保护应用完成")
    
    # 开始训练
    logger.info("开始训练...")
    trainer_wrapper.trainer.train()
    
    # 保存模型
    logger.info("保存模型...")
    trainer_wrapper.trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"模型已保存到: {config.output_dir}")
    
    # 输出隐私信息
    if config.use_dp and trainer_wrapper.privacy_engine:
        epsilon_spent = trainer_wrapper.get_epsilon_spent(config.target_delta)
        logger.info(f"隐私消耗: ε = {epsilon_spent:.4f}, δ = {config.target_delta}")

if __name__ == "__main__":
    main()
# main.py
"""
使用传统差分隐私的主程序入口
"""
from config.training_config import TrainingConfig
from data import MedicalDialogueDataset, DataProcessor
from model import create_lora_model, ModelUtils, ModelMerger
from privacy import BasicDPNoiseCalculator
from trainer import DPTrainer, Evaluator
import torch

def main():
    """主函数"""
    # 加载配置
    config = TrainingConfig()
    
    # 创建模型和分词器
    print("创建LoRA模型...")
    model, tokenizer = create_lora_model(config.MODEL_PATH, config)  # 传递配置参数
    
    # 打印模型信息
    ModelUtils.print_model_info(model)
    
    # 加载训练数据
    print("加载训练数据...")
    full_dataset = MedicalDialogueDataset(config.TRAIN_DATA_PATH, tokenizer, config.MAX_LENGTH)
    
    # 分割数据
    print("分割训练数据...")
    data_processor = DataProcessor()
    train_dataset, val_dataset = data_processor.split_train_data(full_dataset, num_parts=5)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 计算采样率
    sample_rate = (config.PER_DEVICE_TRAIN_BATCH_SIZE * 
                   config.GRADIENT_ACCUMULATION_STEPS / 
                   len(train_dataset))
    
    # 计算总的训练步数
    steps_per_epoch = len(train_dataset) // (config.PER_DEVICE_TRAIN_BATCH_SIZE * 
                                           config.GRADIENT_ACCUMULATION_STEPS)
    total_steps = steps_per_epoch * config.NUM_TRAIN_EPOCHS
    
    print(f"采样率: {sample_rate:.6f}")
    print(f"总训练步数: {total_steps}")
    
    # 使用传统差分隐私计算噪声参数
    print("计算传统差分隐私参数...")
    
    # 计算噪声乘数
    noise_multiplier = BasicDPNoiseCalculator.compute_noise_multiplier(
        target_epsilon=config.TARGET_EPSILON,
        target_delta=config.TARGET_DELTA,
        max_grad_norm=config.MAX_GRAD_NORM
    )
    
    # 计算放大后的epsilon
    amplified_epsilon = BasicDPNoiseCalculator.amplify_epsilon(
        config.TARGET_EPSILON, sample_rate
    )
    
    print(f"计算得到的噪声乘数: {noise_multiplier:.4f}")
    print(f"原始目标隐私: ε={config.TARGET_EPSILON}, δ={config.TARGET_DELTA}")
    print(f"采样率放大后epsilon: {amplified_epsilon:.4f}")
    
    # 初始化训练器
    dp_trainer = DPTrainer(config)
    
    # 执行训练
    print("开始训练...")
    trainer = dp_trainer.train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        noise_multiplier=noise_multiplier
    )
    
    # 计算最终隐私消耗
    epsilon_spent = dp_trainer.get_epsilon_spent(config.TARGET_DELTA)
    print(f"最终消耗的隐私: ε = {epsilon_spent:.2f}")
    
    # 保存模型
    print("保存模型...")
    model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    
    # 评估模型
    print("评估模型...")
    from data import PrivacyTestDataset
    test_dataset = PrivacyTestDataset(config.VALID_DATA_PATH, tokenizer, config.MAX_LENGTH)
    evaluator = Evaluator()
    evaluator.evaluate_model(model, tokenizer, test_dataset)
    
    # 合并LoRA模型
    print("合并LoRA模型...")
    ModelMerger.merge_lora_model(
        base_model_path=config.MODEL_PATH,
        lora_adapter_path=config.OUTPUT_DIR,
        merged_model_path=config.MERGED_MODEL_PATH
    )
    
    print("训练完成!")

if __name__ == "__main__":
    main()
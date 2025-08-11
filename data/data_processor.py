# data/data_processor.py
"""
数据处理和分割
"""
import torch
from torch.utils.data import random_split
from .medical_dataset import MedicalDialogueDataset

class DataProcessor:
    """数据处理器"""
    
    @staticmethod
    def split_train_data(dataset, num_parts: int = 5):
        """
        将训练数据分割为指定份数
        
        Args:
            dataset: 原始数据集
            num_parts: 分割份数
            
        Returns:
            train_dataset, val_dataset: 训练集和验证集
        """
        dataset_size = len(dataset)
        part_size = dataset_size // num_parts
        part_sizes = [part_size] * (num_parts - 1) + [dataset_size - (num_parts - 1) * part_size]
        
        # 分割数据集
        parts = random_split(dataset, part_sizes, generator=torch.Generator().manual_seed(42))
        
        # 使用前num_parts-1份作为训练集，最后一份作为验证集
        train_datasets = parts[:-1]
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        val_dataset = parts[-1]
        
        return train_dataset, val_dataset
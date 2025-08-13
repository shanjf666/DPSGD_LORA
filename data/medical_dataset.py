# data/medical_dataset.py
"""
医疗对话数据集处理
"""
import json
from typing import List, Dict
from torch.utils.data import Dataset
import torch

class MedicalDialogueDataset(Dataset):
    """医疗对话数据集类"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                # 将对话格式化为连续文本
                conversation_text = self._format_conversation(item['conversations'])
                self.data.append(conversation_text)
    
    def _format_conversation(self, conversations: List[Dict]) -> str:
        """将对话格式化为连续文本"""
        formatted = ""
        for turn in conversations:
            if turn['from'] == 'human':
                formatted += f"Human: {turn['value']}\n"
            else:
                formatted += f"Assistant: {turn['value']}\n"
        return formatted.strip()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }


class PrivacyTestDataset(Dataset):
    """隐私测试数据集类"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.privacy_data = []
        self.accuracy_data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                conversation_text = self._format_conversation(item['conversations'])
                if 'name_key' in item and item['name_key'].startswith('privacy_'):
                    self.privacy_data.append(conversation_text)
                else:
                    self.accuracy_data.append(conversation_text)
    
    def _format_conversation(self, conversations: List[Dict]) -> str:
        formatted = ""
        for turn in conversations:
            if turn['from'] == 'human':
                formatted += f"Human: {turn['value']}\n"
            else:
                formatted += f"Assistant: {turn['value']}\n"
        return formatted.strip()
    
    def get_privacy_test_data(self):
        return self.privacy_data
    
    def get_accuracy_test_data(self):
        return self.accuracy_data
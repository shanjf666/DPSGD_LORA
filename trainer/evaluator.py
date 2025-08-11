# trainer/evaluator.py
"""
模型评估器
"""
import torch

class Evaluator:
    """模型评估器"""
    
    @staticmethod
    def evaluate_model(model, tokenizer, test_dataset, max_samples=10):
        """
        评估模型
        
        Args:
            model: 模型
            tokenizer: 分词器
            test_dataset: 测试数据集
            max_samples: 最大评估样本数
        """
        privacy_test_data = test_dataset.get_privacy_test_data()
        accuracy_test_data = test_dataset.get_accuracy_test_data()
        
        print(f"隐私测试样本数: {len(privacy_test_data)}")
        print(f"准确性测试样本数: {len(accuracy_test_data)}")
        
        # 对于准确性测试: 评估模型性能
        model.eval()
        total_predictions = min(max_samples, len(accuracy_test_data))
        
        print("样本预测:")
        for i in range(total_predictions):
            if i < len(accuracy_test_data):
                prompt = accuracy_test_data[i]
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs['input_ids'].to(model.device),
                        max_new_tokens=100,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7
                    )
                    
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"输入: {prompt[:100]}...")
                    print(f"生成: {generated_text[len(prompt):100]}...")
                    print("-" * 50)
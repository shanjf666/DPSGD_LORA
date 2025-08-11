# model/model_utils.py
"""
模型工具类
"""

class ModelUtils:
    """模型工具类"""
    
    @staticmethod
    def count_parameters(model):
        """
        计算模型参数数量
        
        Args:
            model: 模型对象
            
        Returns:
            total_params, trainable_params: 总参数和可训练参数数量
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return total_params, trainable_params
    
    @staticmethod
    def print_model_info(model):
        """
        打印模型信息
        
        Args:
            model: 模型对象
        """
        total_params, trainable_params = ModelUtils.count_parameters(model)
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        print(f"参数压缩率: {trainable_params/total_params*100:.2f}%")
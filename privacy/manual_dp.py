# privacy/manual_dp.py
import torch
import math

class PrivacyEngine:
    
    def __init__(self, noise_multiplier, max_grad_norm):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.steps = 0
        
    def clip_gradients(self, model):
        """裁剪模型梯度"""
        if self.max_grad_norm <= 0:
            return

        //对整个批次梯度的L2范数总和进行裁剪
        //相对于逐样本梯度裁剪，这个方法牺牲了一定的隐私保护但计算效率高
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)   //计算所有参数梯度的L2范数总和
        
        if total_norm > self.max_grad_norm:
            clip_coef = self.max_grad_norm / (total_norm + 1e-6)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
    
    def add_noise(self, model):
        """向梯度添加噪声"""
        if self.noise_multiplier <= 0:
            return
            
        for p in model.parameters():
            if p.grad is not None:
                noise_std = self.noise_multiplier * self.max_grad_norm
                noise = torch.normal(
                    mean=0,
                    std=noise_std,
                    size=p.grad.data.size(),
                    device=p.grad.device
                )
                p.grad.data.add_(noise)
    
    def step(self, model):
        """执行一步训练：裁剪梯度并添加噪声"""
        self.clip_gradients(model)
        self.add_noise(model)
        self.steps += 1
    
    def get_epsilon(self, delta):
        
        if self.noise_multiplier <= 0 or self.steps <= 0:
            return 0.0
            
        # 使用高级组合定理 (Advanced Composition)
        # ε_total ≈ sqrt(2 * T * log(1/δ')) * ε + T * ε * (exp(ε) - 1)
        # 其中 δ' = δ / T, ε = 1/noise_multiplier (每个步骤的epsilon)
        
        epsilon_per_step = 1.0 / self.noise_multiplier
        delta_per_step = delta / self.steps if self.steps > 0 else delta
        
        # 避免数值问题
        if epsilon_per_step > 1.0:
            # 简化计算
            return self.steps * epsilon_per_step * 2
        
        try:
            term1 = math.sqrt(2 * self.steps * math.log(1.0 / delta_per_step)) * epsilon_per_step
            term2 = self.steps * epsilon_per_step * (math.exp(epsilon_per_step) - 1)
            return term1 + term2
        except (OverflowError, ValueError):
            # 如果计算溢出，使用简化公式
            return self.steps * epsilon_per_step * 2

def make_private(model, optimizer, data_loader, noise_multiplier, max_grad_norm):
    """
    创建手动私有化训练组件
    
    Args:
        model: 模型
        optimizer: 优化器
        data_loader: 数据加载器
        noise_multiplier: 噪声乘数
        max_grad_norm: 最大梯度范数
        
    Returns:
        private_model, private_optimizer, private_data_loader, privacy_engine
    """
    privacy_engine = PrivacyEngine(noise_multiplier, max_grad_norm)
    return model, optimizer, data_loader, privacy_engine
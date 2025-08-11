# privacy/dp_calculator.py
"""
传统差分隐私噪声计算器
"""
import math

class BasicDPNoiseCalculator:
    
    @staticmethod
    def compute_noise_std(
        target_epsilon: float,
        target_delta: float,
        max_grad_norm: float
    ) -> float:
        """
        计算满足传统差分隐私的噪声标准差
        
        Args:
            target_epsilon: 目标epsilon值
            target_delta: 目标delta值
            max_grad_norm: 最大梯度范数（L2敏感度）
            
        Returns:
            noise_std: 噪声标准差
        """
        # 使用高斯机制的噪声标准差计算公式
        noise_std = (max_grad_norm * math.sqrt(2 * math.log(1.25 / target_delta))) / target_epsilon
        return noise_std
    
    @staticmethod
    def compute_noise_multiplier(
        target_epsilon: float,
        target_delta: float,
        max_grad_norm: float
    ) -> float:
        """
        计算噪声乘数（与Opacus兼容）
        
        Args:
            target_epsilon: 目标epsilon值
            target_delta: 目标delta值
            max_grad_norm: 最大梯度范数
            
        Returns:
            noise_multiplier: 噪声乘数
        """
        noise_std = BasicDPNoiseCalculator.compute_noise_std(
            target_epsilon, target_delta, max_grad_norm
        )
        # 噪声乘数 = 噪声标准差 / 梯度裁剪范数
        noise_multiplier = noise_std / max_grad_norm
        return noise_multiplier
    
    @staticmethod
    def amplify_epsilon(
        epsilon: float,
        sample_rate: float
    ) -> float:
        """
        根据采样率放大epsilon（隐私放大）
        
        Args:
            epsilon: 原始epsilon
            sample_rate: 采样率
            
        Returns:
            amplified_epsilon: 放大后的epsilon
        """
        return epsilon * sample_rate
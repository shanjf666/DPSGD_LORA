# privacy/__init__.py
from .dp_calculator import BasicDPNoiseCalculator
# 注释掉对Opacus的依赖
# from .privacy_engine import BasicPrivacyEngineWrapper

__all__ = [
    'BasicDPNoiseCalculator',
    # 'BasicPrivacyEngineWrapper'  # 移除这个依赖
]
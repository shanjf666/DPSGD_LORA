# utils/logger.py
"""
日志工具
"""
import logging

def setup_logger(name, log_file, level=logging.INFO):
    """
    设置日志记录器
    
    Args:
        name: 日志名称
        log_file: 日志文件路径
        level: 日志级别
        
    Returns:
        logger: 日志记录器
    """
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger
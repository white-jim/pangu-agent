"""
pangu_agent包
基于OpenPangu大模型的问答智能体
"""
from .agent import PanguAgent
from .config import Settings, get_settings

__version__ = "1.0.0"

__all__ = [
    "PanguAgent",
    "Settings",
    "get_settings"
]

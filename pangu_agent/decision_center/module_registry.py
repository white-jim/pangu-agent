"""
模块注册发现机制
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ModuleStatus(Enum):
    """模块状态"""
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class ModuleInfo:
    """模块信息"""
    name: str
    description: str
    version: str = "1.0.0"
    status: ModuleStatus = ModuleStatus.READY
    supported_intents: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "status": self.status.value,
            "supported_intents": self.supported_intents,
            "capabilities": self.capabilities
        }


@dataclass
class ModuleResult:
    """模块执行结果"""
    success: bool
    data: Any
    message: str = ""
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "data": self.data,
            "message": self.message,
            "metadata": self.metadata
        }


class BaseModule(ABC):
    """
    模块基类
    所有功能模块必须继承此类
    """
    
    def __init__(self):
        self._status = ModuleStatus.READY
        self._info: Optional[ModuleInfo] = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """模块名称"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """模块描述"""
        pass
    
    @property
    def version(self) -> str:
        """模块版本"""
        return "1.0.0"
    
    @property
    def supported_intents(self) -> List[str]:
        """支持的意图类型"""
        return []
    
    @property
    def capabilities(self) -> List[str]:
        """模块能力列表"""
        return []
    
    def get_info(self) -> ModuleInfo:
        """获取模块信息"""
        if self._info is None:
            self._info = ModuleInfo(
                name=self.name,
                description=self.description,
                version=self.version,
                status=self._status,
                supported_intents=self.supported_intents,
                capabilities=self.capabilities
            )
        return self._info
    
    def get_status(self) -> ModuleStatus:
        """获取模块状态"""
        return self._status
    
    def set_status(self, status: ModuleStatus):
        """设置模块状态"""
        self._status = status
        if self._info:
            self._info.status = status
    
    @abstractmethod
    def execute(self, query: str, context: Dict, **kwargs) -> ModuleResult:
        """
        执行模块功能
        
        Args:
            query: 用户查询
            context: 上下文信息
            **kwargs: 其他参数
            
        Returns:
            ModuleResult: 执行结果
        """
        pass
    
    def initialize(self) -> bool:
        """
        初始化模块
        子类可重写此方法进行初始化操作
        
        Returns:
            bool: 初始化是否成功
        """
        logger.info(f"模块 {self.name} 初始化完成")
        return True
    
    def shutdown(self):
        """
        关闭模块
        子类可重写此方法进行清理操作
        """
        logger.info(f"模块 {self.name} 已关闭")


class ModuleRegistry:
    """
    模块注册中心
    管理所有功能模块的注册、发现和调度
    """
    
    _instance: Optional['ModuleRegistry'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._modules: Dict[str, BaseModule] = {}
            cls._instance._intent_mapping: Dict[str, str] = {}
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'ModuleRegistry':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def register(self, module: BaseModule) -> bool:
        """
        注册模块
        
        Args:
            module: 模块实例
            
        Returns:
            bool: 注册是否成功
        """
        if module.name in self._modules:
            logger.warning(f"模块 {module.name} 已注册，将被覆盖")
        
        self._modules[module.name] = module
        
        for intent in module.supported_intents:
            self._intent_mapping[intent] = module.name
        
        logger.info(f"模块 {module.name} 注册成功，支持意图: {module.supported_intents}")
        return True
    
    def unregister(self, module_name: str) -> bool:
        """
        注销模块
        
        Args:
            module_name: 模块名称
            
        Returns:
            bool: 注销是否成功
        """
        if module_name not in self._modules:
            logger.warning(f"模块 {module_name} 未注册")
            return False
        
        module = self._modules[module_name]
        
        for intent in module.supported_intents:
            if intent in self._intent_mapping:
                del self._intent_mapping[intent]
        
        del self._modules[module_name]
        logger.info(f"模块 {module_name} 已注销")
        return True
    
    def get_module(self, module_name: str) -> Optional[BaseModule]:
        """
        获取模块
        
        Args:
            module_name: 模块名称
            
        Returns:
            Optional[BaseModule]: 模块实例
        """
        return self._modules.get(module_name)
    
    def get_module_by_intent(self, intent: str) -> Optional[BaseModule]:
        """
        根据意图获取模块
        
        Args:
            intent: 意图类型
            
        Returns:
            Optional[BaseModule]: 模块实例
        """
        module_name = self._intent_mapping.get(intent)
        if module_name:
            return self._modules.get(module_name)
        return None
    
    def list_modules(self) -> List[ModuleInfo]:
        """列出所有已注册模块"""
        return [module.get_info() for module in self._modules.values()]
    
    def get_all_intents(self) -> Dict[str, str]:
        """获取所有意图到模块的映射"""
        return self._intent_mapping.copy()
    
    def initialize_all(self) -> Dict[str, bool]:
        """初始化所有模块"""
        results = {}
        for name, module in self._modules.items():
            try:
                results[name] = module.initialize()
            except Exception as e:
                logger.error(f"模块 {name} 初始化失败: {e}")
                results[name] = False
        return results
    
    def shutdown_all(self):
        """关闭所有模块"""
        for module in self._modules.values():
            try:
                module.shutdown()
            except Exception as e:
                logger.error(f"模块 {module.name} 关闭失败: {e}")

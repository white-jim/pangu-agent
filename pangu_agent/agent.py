"""
问答智能体主类
整合所有模块
"""
import logging
from typing import Dict, Optional, Any

from pangu_agent.config import get_settings
from pangu_agent.models import PanguModel, ExternalAPIClient
from pangu_agent.decision_center import DecisionCenter, BaseModule
from pangu_agent.modules import (
    SemanticInteractionModule,
    NavigationModule,
    VisionModule,
    InteractionModule
)

logger = logging.getLogger(__name__)


class PanguAgent:
    """
    基于OpenPangu大模型的问答智能体
    
    整合模型、决策中枢和各功能模块
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_external_api: bool = False,
        external_api_key: Optional[str] = None,
        external_api_base: Optional[str] = None,
        external_model_name: str = "deepseek-ai/DeepSeek-V3.2",
        intent_threshold: float = 0.6,
        max_context_turns: int = 5,
        rag_top_k: int = 3
    ):
        """
        初始化智能体
        
        Args:
            model_path: OpenPangu模型路径
            use_external_api: 是否使用外部API
            external_api_key: 外部API密钥
            external_api_base: 外部API地址
            external_model_name: 外部模型名称
            intent_threshold: 意图识别阈值
            max_context_turns: 最大上下文轮数
            rag_top_k: RAG检索数量
        """
        self.settings = get_settings()
        
        self._model_path = model_path or self.settings.model.model_path
        self._use_external_api = use_external_api
        self._external_api_key = external_api_key
        self._external_api_base = external_api_base
        self._external_model_name = external_model_name
        self._intent_threshold = intent_threshold
        self._max_context_turns = max_context_turns
        self._rag_top_k = rag_top_k
        
        self._model: Optional[PanguModel] = None
        self._external_api: Optional[ExternalAPIClient] = None
        self._decision_center: Optional[DecisionCenter] = None
        
        self._semantic_module: Optional[SemanticInteractionModule] = None
        self._navigation_module: Optional[NavigationModule] = None
        self._vision_module: Optional[VisionModule] = None
        self._interaction_module: Optional[InteractionModule] = None
        
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        初始化智能体
        
        Returns:
            bool: 初始化是否成功
        """
        if self._initialized:
            logger.info("智能体已初始化")
            return True
        
        logger.info("正在初始化问答智能体...")
        
        try:
            self._init_model()
            
            self._init_modules()
            
            self._init_decision_center()
            
            self._initialized = True
            logger.info("问答智能体初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"智能体初始化失败: {e}")
            return False
    
    def _init_model(self):
        """初始化模型"""
        if self._use_external_api:
            logger.info("使用外部API模式")
            # 优先使用传入的参数，否则使用环境变量配置
            api_key = self._external_api_key or self.settings.external_api.api_key
            api_base = self._external_api_base or self.settings.external_api.api_base
            model_name = self._external_model_name or self.settings.external_api.model_name
            
            logger.info(f"API配置: key={'已设置' if api_key else '未设置'}, base={api_base}, model={model_name}")
            
            if not api_key:
                raise ValueError("使用外部API需要提供API密钥，请设置EXTERNAL_API_KEY环境变量或传入api_key参数")
            
            self._external_api = ExternalAPIClient(
                api_key=api_key,
                api_base=api_base,
                model_name=model_name
            )
            logger.info("外部API客户端初始化完成")
        elif self._model_path:
            logger.info(f"加载OpenPangu模型: {self._model_path}")
            self._model = PanguModel(
                model_path=self._model_path,
                device=self.settings.model.device,
                max_length=self.settings.model.max_length,
                temperature=self.settings.model.temperature,
                top_p=self.settings.model.top_p,
                trust_remote_code=self.settings.model.trust_remote_code
            )
            self._model.load_model()
        else:
            logger.warning("未配置模型路径，将使用RAG检索模式（无大模型生成）")
    
    def _init_modules(self):
        """初始化功能模块"""
        self._semantic_module = SemanticInteractionModule(
            top_k=self._rag_top_k
        )
        self._semantic_module.initialize()
        
        model_client = self._model or self._external_api
        if model_client:
            self._semantic_module.set_model_client(model_client)
        
        self._navigation_module = NavigationModule()
        self._navigation_module.initialize()
        
        self._vision_module = VisionModule()
        self._vision_module.initialize()
        
        self._interaction_module = InteractionModule()
        self._interaction_module.initialize()
        
        logger.info("功能模块初始化完成")
    
    def _init_decision_center(self):
        """初始化决策中枢"""
        model_client = self._model or self._external_api
        
        self._decision_center = DecisionCenter(
            intent_threshold=self._intent_threshold,
            max_context_turns=self._max_context_turns,
            model_client=model_client
        )
        
        self._decision_center.register_module(self._semantic_module)
        self._decision_center.register_module(self._navigation_module)
        self._decision_center.register_module(self._vision_module)
        
        self._decision_center.initialize()
        
        logger.info("决策中枢初始化完成")
    
    def process(self, query: str, session_id: str = "default") -> Dict:
        """
        处理用户查询
        
        Args:
            query: 用户查询
            session_id: 会话ID
            
        Returns:
            Dict: 处理结果
        """
        if not self._initialized:
            raise RuntimeError("智能体未初始化，请先调用initialize()")
        
        result = self._decision_center.process(query, session_id)
        return result.to_dict()
    
    def add_knowledge(self, documents: list) -> int:
        """
        添加知识到知识库
        
        Args:
            documents: 文档列表
            
        Returns:
            int: 添加的文档数量
        """
        if self._semantic_module:
            return self._semantic_module.add_knowledge(documents)
        return 0
    
    def clear_session(self, session_id: str):
        """清除会话"""
        if self._decision_center:
            self._decision_center.clear_session(session_id)
    
    def get_status(self) -> Dict:
        """获取系统状态"""
        status = {
            "initialized": self._initialized,
            "model": {
                "type": "external_api" if self._use_external_api else "local",
                "loaded": self._model.is_loaded() if self._model else False
            },
            "modules": {}
        }
        
        if self._decision_center:
            status["decision_center"] = self._decision_center.get_status()
        
        if self._semantic_module:
            status["modules"]["semantic"] = self._semantic_module.get_stats()
        
        return status
    
    def shutdown(self):
        """关闭智能体"""
        logger.info("关闭问答智能体...")
        
        if self._decision_center:
            self._decision_center.shutdown()
        
        self._initialized = False
        logger.info("问答智能体已关闭")

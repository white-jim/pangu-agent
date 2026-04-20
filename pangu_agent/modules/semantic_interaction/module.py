"""
通用语义交互模块
基于RAG的问答功能
"""
import logging
from typing import Dict, List, Optional, Any

from pangu_agent.decision_center import BaseModule, ModuleResult
from .rag_engine import RAGEngine
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class SemanticInteractionModule(BaseModule):
    """
    通用语义交互模块
    实现基于RAG的问答功能
    """
    
    @property
    def name(self) -> str:
        return "semantic_interaction"
    
    @property
    def description(self) -> str:
        return "通用语义问答模块，基于RAG技术实现知识检索和回答生成"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_intents(self) -> List[str]:
        return ["qa", "registration"]
    
    @property
    def capabilities(self) -> List[str]:
        return [
            "knowledge_retrieval",
            "question_answering",
            "context_aware_response"
        ]
    
    def __init__(
        self,
        top_k: int = 3,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        super().__init__()
        
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self._vector_store: Optional[VectorStore] = None
        self._rag_engine: Optional[RAGEngine] = None
        self._model_client = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """初始化模块"""
        if self._initialized:
            return True
        
        try:
            logger.info("初始化语义交互模块...")
            
            self._vector_store = VectorStore()
            
            self._rag_engine = RAGEngine(
                vector_store=self._vector_store,
                top_k=self.top_k,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            self._load_default_knowledge()
            
            self._initialized = True
            logger.info("语义交互模块初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"语义交互模块初始化失败: {e}")
            return False
    
    def set_model_client(self, client: Any):
        """设置模型客户端"""
        self._model_client = client
        if self._rag_engine:
            self._rag_engine.set_model_client(client)
    
    def _load_default_knowledge(self):
        """加载默认知识库"""
        default_knowledge = [
            {
                "content": "本项目是基于OpenPangu大模型的问答智能体系统，采用'智能决策中枢 + 分布式功能模块'架构设计。",
                "metadata": {"source": "项目介绍", "category": "overview"}
            },
            {
                "content": "OpenPangu大模型是本系统的核心大脑，负责理解用户意图、调度各个功能模块、生成最终回答。模型支持在昇腾NPU上运行。",
                "metadata": {"source": "项目介绍", "category": "model"}
            },
            {
                "content": "智能决策中枢负责意图识别、模块调度、多轮对话上下文管理和结果整合。它能判断用户问题是问答、导航、视觉还是注册等类型。",
                "metadata": {"source": "项目介绍", "category": "decision"}
            },
            {
                "content": "通用语义交互模块使用RAG（检索增强生成）技术，通过向量数据库进行知识检索，结合大模型生成准确回答。",
                "metadata": {"source": "项目介绍", "category": "rag"}
            },
            {
                "content": "导航导览模块负责室内导航和路径规划，可以引导用户到达指定位置。该功能目前处于开发阶段。",
                "metadata": {"source": "项目介绍", "category": "navigation"}
            },
            {
                "content": "视觉交互模块支持人脸检测和识别功能，可以识别用户身份。该功能目前处于开发阶段。",
                "metadata": {"source": "项目介绍", "category": "vision"}
            },
            {
                "content": "本系统支持多轮对话，能够记住上下文信息，提供连贯的交互体验。",
                "metadata": {"source": "项目介绍", "category": "dialogue"}
            },
            {
                "content": "系统使用Gradio构建Web演示界面，用户可以通过浏览器与智能体进行交互。",
                "metadata": {"source": "项目介绍", "category": "interface"}
            }
        ]
        
        if self._rag_engine:
            count = self._rag_engine.add_knowledge(default_knowledge)
            logger.info(f"加载默认知识库: {count} 条知识")
    
    def add_knowledge(self, documents: List[Dict[str, Any]]) -> int:
        """
        添加知识到知识库
        
        Args:
            documents: 文档列表
            
        Returns:
            int: 添加的文档数量
        """
        if not self._rag_engine:
            logger.warning("RAG引擎未初始化")
            return 0
        
        return self._rag_engine.add_knowledge(documents)
    
    def execute(self, query: str, context: Dict, **kwargs) -> ModuleResult:
        """执行问答，将历史对话和工具链中间结果传入 RAG 引擎。"""
        if not self._initialized:
            return ModuleResult(
                success=False,
                data=None,
                message="模块未初始化"
            )

        try:
            history = context.get("history") or []
            intermediate_results = context.get("intermediate_results") or {}

            rag_result = self._rag_engine.query(
                query,
                top_k=self.top_k,
                history=history if history else None,
                intermediate_results=intermediate_results if intermediate_results else None
            )

            retrieved_info = [
                {
                    "content": doc.document.content,
                    "score": doc.score,
                    "metadata": doc.document.metadata
                }
                for doc in rag_result.retrieved_docs
            ]

            return ModuleResult(
                success=True,
                data=rag_result.generated_answer,
                message="问答成功",
                metadata={
                    "retrieved_docs": retrieved_info,
                    "doc_count": len(rag_result.retrieved_docs),
                    "prompt_used": rag_result.prompt_used,
                    "history_turns": len(history),
                    "has_intermediate_results": bool(intermediate_results)
                }
            )

        except Exception as e:
            logger.error(f"问答执行失败: {e}")
            return ModuleResult(
                success=False,
                data=None,
                message=f"问答处理失败: {str(e)}"
            )
    
    def get_stats(self) -> Dict:
        """获取模块统计信息"""
        if not self._rag_engine:
            return {"initialized": False}
        
        stats = self._rag_engine.get_stats()
        stats["initialized"] = self._initialized
        return stats
    
    def shutdown(self):
        """关闭模块"""
        logger.info("关闭语义交互模块")
        self._initialized = False

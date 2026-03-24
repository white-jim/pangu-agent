"""
多轮对话上下文管理
"""
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DialogueTurn:
    """单轮对话"""
    turn_id: int
    user_input: str
    intent: str
    module_name: str
    response: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "turn_id": self.turn_id,
            "user_input": self.user_input,
            "intent": self.intent,
            "module_name": self.module_name,
            "response": self.response,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class DialogueContext:
    """对话上下文"""
    session_id: str
    turns: List[DialogueTurn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)
    
    def add_turn(self, turn: DialogueTurn):
        """添加一轮对话"""
        self.turns.append(turn)
        self.updated_at = time.time()
    
    def get_last_n_turns(self, n: int) -> List[DialogueTurn]:
        """获取最近n轮对话"""
        return self.turns[-n:] if n > 0 else []
    
    def get_context_text(self, max_turns: int = 5) -> str:
        """获取上下文文本"""
        turns = self.get_last_n_turns(max_turns)
        context_parts = []
        for turn in turns:
            context_parts.append(f"用户: {turn.user_input}")
            context_parts.append(f"助手: {turn.response}")
        return "\n".join(context_parts)
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "turns": [t.to_dict() for t in self.turns],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata
        }


class ContextManager:
    """
    上下文管理器
    管理多轮对话的上下文
    """
    
    def __init__(self, max_context_turns: int = 5):
        self.max_context_turns = max_context_turns
        self._contexts: Dict[str, DialogueContext] = {}
        self._turn_counter: Dict[str, int] = defaultdict(int)
    
    def create_context(self, session_id: str) -> DialogueContext:
        """
        创建新的对话上下文
        
        Args:
            session_id: 会话ID
            
        Returns:
            DialogueContext: 新创建的上下文
        """
        context = DialogueContext(session_id=session_id)
        self._contexts[session_id] = context
        self._turn_counter[session_id] = 0
        logger.info(f"创建新会话: {session_id}")
        return context
    
    def get_context(self, session_id: str) -> Optional[DialogueContext]:
        """
        获取对话上下文
        
        Args:
            session_id: 会话ID
            
        Returns:
            Optional[DialogueContext]: 对话上下文
        """
        return self._contexts.get(session_id)
    
    def get_or_create_context(self, session_id: str) -> DialogueContext:
        """
        获取或创建对话上下文
        
        Args:
            session_id: 会话ID
            
        Returns:
            DialogueContext: 对话上下文
        """
        context = self.get_context(session_id)
        if context is None:
            context = self.create_context(session_id)
        return context
    
    def add_turn(
        self,
        session_id: str,
        user_input: str,
        intent: str,
        module_name: str,
        response: str,
        metadata: Optional[Dict] = None
    ) -> DialogueTurn:
        """
        添加一轮对话
        
        Args:
            session_id: 会话ID
            user_input: 用户输入
            intent: 意图类型
            module_name: 处理模块名称
            response: 系统回复
            metadata: 元数据
            
        Returns:
            DialogueTurn: 对话轮次
        """
        context = self.get_or_create_context(session_id)
        
        self._turn_counter[session_id] += 1
        turn = DialogueTurn(
            turn_id=self._turn_counter[session_id],
            user_input=user_input,
            intent=intent,
            module_name=module_name,
            response=response,
            metadata=metadata or {}
        )
        
        context.add_turn(turn)
        logger.debug(f"会话 {session_id} 添加第 {turn.turn_id} 轮对话")
        
        return turn
    
    def get_context_for_model(self, session_id: str) -> List[Dict[str, str]]:
        """
        获取用于模型推理的上下文
        
        Args:
            session_id: 会话ID
            
        Returns:
            List[Dict]: 模型格式的对话历史
        """
        context = self.get_context(session_id)
        if context is None:
            return []
        
        messages = []
        for turn in context.get_last_n_turns(self.max_context_turns):
            messages.append({"role": "user", "content": turn.user_input})
            messages.append({"role": "assistant", "content": turn.response})
        
        return messages
    
    def clear_context(self, session_id: str):
        """清除对话上下文"""
        if session_id in self._contexts:
            del self._contexts[session_id]
        if session_id in self._turn_counter:
            del self._turn_counter[session_id]
        logger.info(f"清除会话: {session_id}")
    
    def get_all_sessions(self) -> List[str]:
        """获取所有会话ID"""
        return list(self._contexts.keys())
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """获取会话信息"""
        context = self.get_context(session_id)
        if context:
            return {
                "session_id": session_id,
                "turn_count": len(context.turns),
                "created_at": context.created_at,
                "updated_at": context.updated_at
            }
        return None

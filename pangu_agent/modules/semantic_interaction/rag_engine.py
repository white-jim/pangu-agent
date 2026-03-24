"""
RAG检索增强生成引擎
"""
import logging
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass

from .vector_store import VectorStore, Document, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """RAG检索结果"""
    query: str
    retrieved_docs: List[SearchResult]
    generated_answer: str
    prompt_used: str
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "retrieved_docs": [doc.to_dict() for doc in self.retrieved_docs],
            "generated_answer": self.generated_answer,
            "prompt_used": self.prompt_used
        }


class RAGEngine:
    """
    RAG检索增强生成引擎
    整合向量检索和大模型生成
    """
    
    DEFAULT_SYSTEM_PROMPT = """你是一个智能问答助手。请根据提供的参考资料回答用户的问题。
如果参考资料中没有相关信息，请根据你的知识给出合理的回答。
回答要准确、简洁、有帮助。"""

    DEFAULT_PROMPT_TEMPLATE = """参考资料：
{context}

用户问题：{question}

请根据以上参考资料回答用户问题："""

    def __init__(
        self,
        vector_store: VectorStore,
        model_client: Optional[Any] = None,
        top_k: int = 3,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        system_prompt: Optional[str] = None,
        prompt_template: Optional[str] = None
    ):
        self.vector_store = vector_store
        self.model_client = model_client
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        
        self._text_splitter: Optional[Callable] = None
    
    def set_model_client(self, client: Any):
        """设置模型客户端"""
        self.model_client = client
    
    def add_knowledge(self, documents: List[Dict[str, Any]]) -> int:
        """
        添加知识到向量库
        
        Args:
            documents: 文档列表，每个文档包含content和metadata
            
        Returns:
            int: 添加的文档数量
        """
        docs = []
        for i, doc_data in enumerate(documents):
            content = doc_data.get("content", "")
            metadata = doc_data.get("metadata", {})
            
            chunks = self._split_text(content)
            
            for j, chunk in enumerate(chunks):
                doc = Document(
                    doc_id=f"doc_{i}_{j}",
                    content=chunk,
                    metadata=metadata
                )
                docs.append(doc)
        
        return self.vector_store.add_documents(docs)
    
    def _split_text(self, text: str) -> List[str]:
        """文本分块"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end < len(text):
                for sep in ['。', '！', '？', '；', '.', '!', '?', ';', '\n']:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start:
                        end = last_sep + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
        
        return chunks
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回数量
            
        Returns:
            List[SearchResult]: 检索结果
        """
        top_k = top_k or self.top_k
        return self.vector_store.search(query, top_k)
    
    def build_prompt(self, query: str, retrieved_docs: List[SearchResult]) -> str:
        """
        构建提示词
        
        Args:
            query: 用户查询
            retrieved_docs: 检索到的文档
            
        Returns:
            str: 构建好的提示词
        """
        context_parts = []
        for i, result in enumerate(retrieved_docs, 1):
            context_parts.append(f"[文档{i}] {result.document.content}")
        
        context = "\n\n".join(context_parts) if context_parts else "无相关参考资料"
        
        prompt = self.prompt_template.format(context=context, question=query)
        
        return prompt
    
    def generate(
        self,
        query: str,
        retrieved_docs: List[SearchResult],
        **kwargs
    ) -> str:
        """
        生成回答
        
        Args:
            query: 用户查询
            retrieved_docs: 检索到的文档
            
        Returns:
            str: 生成的回答
        """
        if self.model_client is None:
            return self._generate_fallback(query, retrieved_docs)
        
        prompt = self.build_prompt(query, retrieved_docs)
        
        try:
            if hasattr(self.model_client, 'generate'):
                response = self.model_client.generate(prompt, **kwargs)
            elif hasattr(self.model_client, 'chat'):
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
                response = self.model_client.chat(messages, **kwargs)
            else:
                return self._generate_fallback(query, retrieved_docs)
            
            return response
            
        except Exception as e:
            logger.error(f"模型生成失败: {e}")
            return self._generate_fallback(query, retrieved_docs)
    
    def _generate_fallback(self, query: str, retrieved_docs: List[SearchResult]) -> str:
        """备选生成方法（无模型时）"""
        if not retrieved_docs:
            return "抱歉，我没有找到相关的知识来回答您的问题。"
        
        response_parts = ["根据知识库检索到以下相关信息：\n"]
        
        for i, result in enumerate(retrieved_docs, 1):
            response_parts.append(f"{i}. {result.document.content}\n")
        
        return "\n".join(response_parts)
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> RAGResult:
        """
        RAG查询
        
        Args:
            query: 用户查询
            top_k: 检索数量
            
        Returns:
            RAGResult: RAG结果
        """
        top_k = top_k or self.top_k
        
        retrieved_docs = self.retrieve(query, top_k)
        
        answer = self.generate(query, retrieved_docs, **kwargs)
        
        prompt = self.build_prompt(query, retrieved_docs)
        
        return RAGResult(
            query=query,
            retrieved_docs=retrieved_docs,
            generated_answer=answer,
            prompt_used=prompt
        )
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "document_count": self.vector_store.get_document_count(),
            "top_k": self.top_k,
            "chunk_size": self.chunk_size,
            "has_model": self.model_client is not None
        }

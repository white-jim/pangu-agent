"""
向量数据库模块
支持FAISS向量存储
"""
import logging
import json
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """文档数据结构"""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata
        }


@dataclass
class SearchResult:
    """检索结果"""
    document: Document
    score: float
    
    def to_dict(self) -> Dict:
        return {
            "document": self.document.to_dict(),
            "score": self.score
        }


class VectorStore:
    """
    向量数据库类
    使用FAISS进行向量存储和检索
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        persist_path: Optional[str] = None
    ):
        self.embedding_dim = embedding_dim
        self.persist_path = persist_path
        
        self._documents: Dict[str, Document] = {}
        self._index = None
        self._embedding_model = None
        
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """初始化嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.embedding_dim = self._embedding_model.get_sentence_embedding_dimension()
            logger.info(f"嵌入模型加载成功，维度: {self.embedding_dim}")
        except ImportError:
            logger.warning("sentence-transformers未安装，将使用简单的TF-IDF作为备选方案")
            self._embedding_model = None
        except Exception as e:
            logger.warning(f"嵌入模型加载失败: {e}，将使用简单模式")
            self._embedding_model = None
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本嵌入向量"""
        if self._embedding_model is not None:
            embeddings = self._embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        else:
            return self._simple_embedding(texts)
    
    def _simple_embedding(self, texts: List[str]) -> List[List[float]]:
        """简单的嵌入方法（备选方案）"""
        import hashlib
        embeddings = []
        for text in texts:
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            embedding = [float(b) / 255.0 for b in hash_bytes]
            while len(embedding) < self.embedding_dim:
                embedding.extend(embedding[:min(len(embedding), self.embedding_dim - len(embedding))])
            embeddings.append(embedding[:self.embedding_dim])
        return embeddings
    
    def _init_faiss_index(self):
        """初始化FAISS索引"""
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.embedding_dim)
            logger.info("FAISS索引初始化成功")
        except ImportError:
            logger.warning("FAISS未安装，将使用简单的余弦相似度检索")
            self._index = None
    
    def add_documents(self, documents: List[Document]) -> int:
        """
        添加文档到向量库
        
        Args:
            documents: 文档列表
            
        Returns:
            int: 添加的文档数量
        """
        if not documents:
            return 0
        
        texts = [doc.content for doc in documents]
        embeddings = self._get_embeddings(texts)
        
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
            self._documents[doc.doc_id] = doc
        
        if self._index is None:
            self._init_faiss_index()
        
        if self._index is not None:
            import numpy as np
            emb_array = np.array(embeddings).astype('float32')
            faiss.normalize_L2(emb_array)
            self._index.add(emb_array)
        
        logger.info(f"添加 {len(documents)} 个文档到向量库")
        return len(documents)
    
    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回top-k结果
            
        Returns:
            List[SearchResult]: 检索结果列表
        """
        if not self._documents:
            return []
        
        query_embedding = self._get_embeddings([query])[0]
        
        if self._index is not None and len(self._documents) > 0:
            return self._faiss_search(query_embedding, top_k)
        else:
            return self._simple_search(query_embedding, top_k)
    
    def _faiss_search(self, query_embedding: List[float], top_k: int) -> List[SearchResult]:
        """使用FAISS进行检索"""
        import numpy as np
        
        query_vec = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vec)
        
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k)
        
        results = []
        doc_list = list(self._documents.values())
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(doc_list):
                results.append(SearchResult(
                    document=doc_list[idx],
                    score=float(score)
                ))
        
        return results
    
    def _simple_search(self, query_embedding: List[float], top_k: int) -> List[SearchResult]:
        """简单的余弦相似度检索"""
        import numpy as np
        
        query_vec = np.array(query_embedding)
        
        scores = []
        doc_list = list(self._documents.values())
        
        for doc in doc_list:
            if doc.embedding:
                doc_vec = np.array(doc.embedding)
                similarity = np.dot(query_vec, doc_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-8
                )
                scores.append((similarity, doc))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        
        return [
            SearchResult(document=doc, score=float(score))
            for score, doc in scores[:top_k]
        ]
    
    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        if doc_id in self._documents:
            del self._documents[doc_id]
            logger.info(f"删除文档: {doc_id}")
            return True
        return False
    
    def clear(self):
        """清空向量库"""
        self._documents.clear()
        if self._index is not None:
            self._init_faiss_index()
        logger.info("向量库已清空")
    
    def get_document_count(self) -> int:
        """获取文档数量"""
        return len(self._documents)
    
    def save(self, path: Optional[str] = None):
        """保存向量库"""
        path = path or self.persist_path
        if not path:
            return
        
        os.makedirs(path, exist_ok=True)
        
        docs_data = [
            {"doc_id": doc.doc_id, "content": doc.content, "metadata": doc.metadata}
            for doc in self._documents.values()
        ]
        
        with open(os.path.join(path, "documents.json"), 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)
        
        if self._index is not None:
            import faiss
            faiss.write_index(self._index, os.path.join(path, "index.faiss"))
        
        logger.info(f"向量库已保存到: {path}")
    
    def load(self, path: Optional[str] = None):
        """加载向量库"""
        path = path or self.persist_path
        if not path or not os.path.exists(path):
            return
        
        docs_path = os.path.join(path, "documents.json")
        if os.path.exists(docs_path):
            with open(docs_path, 'r', encoding='utf-8') as f:
                docs_data = json.load(f)
            
            documents = [
                Document(
                    doc_id=doc["doc_id"],
                    content=doc["content"],
                    metadata=doc["metadata"]
                )
                for doc in docs_data
            ]
            
            self.add_documents(documents)
        
        index_path = os.path.join(path, "index.faiss")
        if os.path.exists(index_path) and self._index is not None:
            import faiss
            self._index = faiss.read_index(index_path)
        
        logger.info(f"向量库已从 {path} 加载")

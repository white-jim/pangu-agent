"""
项目配置文件
"""
import os
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# 加载.env文件
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)


@dataclass
class ModelConfig:
    model_path: str = ""
    model_name: str = "pangu"
    device: str = "npu"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    trust_remote_code: bool = True


@dataclass
class ExternalAPIConfig:
    enabled: bool = False
    api_type: str = "openai"
    api_key: str = ""
    api_base: str = ""
    model_name: str = "gpt-3.5-turbo"


@dataclass
class RAGConfig:
    vector_db_type: str = "faiss"
    embedding_model: str = "text2vec-base-chinese"
    top_k: int = 3
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass
class DecisionConfig:
    intent_threshold: float = 0.6
    max_context_turns: int = 5


@dataclass
class Settings:
    project_name: str = "基于OpenPangu大模型的问答智能体"
    version: str = "1.0.0"
    
    model: ModelConfig = field(default_factory=ModelConfig)
    external_api: ExternalAPIConfig = field(default_factory=ExternalAPIConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(init=False)
    knowledge_file: Path = field(init=False)
    
    def __post_init__(self):
        self.data_dir = self.base_dir / "data"
        self.knowledge_file = self.data_dir / "knowledge.json"
        
        self.model.model_path = os.getenv("PANGU_MODEL_PATH", "")
        self.external_api.api_key = os.getenv("EXTERNAL_API_KEY", "")
        self.external_api.api_base = os.getenv("EXTERNAL_API_BASE", "")
        self.external_api.model_name = os.getenv("EXTERNAL_MODEL_NAME", self.external_api.model_name)
        self.external_api.enabled = os.getenv("USE_EXTERNAL_API", "false").lower() == "true"


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

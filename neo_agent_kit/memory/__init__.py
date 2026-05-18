"""neo-agent-kit 记忆系统模块

提供完整的记忆系统实现:
- WorkingMemory: 工作记忆（临时信息，TTL管理）
- EpisodicMemory: 情景记忆（事件序列，SQLite持久化）
- SemanticMemory: 语义记忆（抽象知识，概念+实体提取）
- MemoryManager: 统一调度和协调
- MemoryConfig: 系统配置
"""
from .base import MemoryItem, MemoryConfig, BaseMemory
from .manager import MemoryManager
from .embedding import create_embedding_model, TFIDFEmbedding, cosine_similarity
from .types import WorkingMemory, EpisodicMemory, SemanticMemory

__all__ = [
    "MemoryItem",
    "MemoryConfig",
    "BaseMemory",
    "MemoryManager",
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "create_embedding_model",
    "TFIDFEmbedding",
    "cosine_similarity",
]

"""记忆系统基础数据结构

提供 MemoryItem, MemoryConfig 和 BaseMemory 抽象基类。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class MemoryItem:
    """标准化的记忆数据结构

    模拟人类记忆中每个记忆单元的基本属性。
    """

    id: str
    content: str
    memory_type: str = "working"  # working / episodic / semantic / perceptual
    importance: float = 0.5  # 0.0-1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class MemoryConfig:
    """记忆系统配置"""

    # 工作记忆
    working_memory_capacity: int = 50
    working_memory_ttl: int = 60  # 分钟

    # 数据库路径
    database_path: str = "./memory_data/memory.db"

    # 嵌入模型配置
    embed_model_type: str = "tfidf"  # tfidf / local / dashscope
    embed_model_name: str = ""
    embed_api_key: str = ""
    embed_base_url: str = ""

    # 向量数据库 (可选)
    use_qdrant: bool = False
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # 图数据库 (可选)
    use_neo4j: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "neo4j"


class BaseMemory(ABC):
    """记忆类型抽象基类

    所有记忆类型 (Working, Episodic, Semantic, Perceptual) 都继承自此类。
    """

    def __init__(self, config: MemoryConfig):
        self.config = config

    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str:
        """添加记忆，返回记忆ID"""
        pass

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索记忆"""
        pass

    @abstractmethod
    def forget(self, strategy: str = "importance_based", **kwargs) -> int:
        """遗忘记忆，返回遗忘数量"""
        pass

    @abstractmethod
    def clear(self) -> int:
        """清空所有记忆，返回清除数量"""
        pass

    @abstractmethod
    def count(self) -> int:
        """获取记忆数量"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        pass

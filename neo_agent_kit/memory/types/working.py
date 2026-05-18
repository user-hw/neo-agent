"""工作记忆 (WorkingMemory) 实现

特点:
- 容量有限（默认50条）+ TTL 自动清理
- 纯内存存储，访问速度极快
- 混合检索：TF-IDF 向量化 + 关键词匹配

模拟人类的工作记忆：短暂、容量有限、用于当前任务处理。
"""
from __future__ import annotations

import time
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List

from ..base import BaseMemory, MemoryConfig, MemoryItem
from ..embedding import TFIDFEmbedding, cosine_similarity


class WorkingMemory(BaseMemory):
    """工作记忆 - 当前会话的临时信息存储"""

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self.max_capacity = config.working_memory_capacity or 50
        self.max_age_seconds = (config.working_memory_ttl or 60) * 60
        self.memories: List[MemoryItem] = []
        self._tfidf = TFIDFEmbedding()

    # ========== 核心操作 ==========

    def add(self, memory_item: MemoryItem) -> str:
        """添加工作记忆"""
        self._expire_old_memories()

        # 容量管理：满了就移除最不重要的
        if len(self.memories) >= self.max_capacity:
            self._remove_lowest_priority()

        memory_item.memory_type = "working"
        self.memories.append(memory_item)
        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """混合检索：TF-IDF 向量化 + 关键词匹配"""
        self._expire_old_memories()

        if not self.memories:
            return []

        min_importance = kwargs.get("min_importance", 0.0)

        try:
            query_vec = self._tfidf.encode(query)[0]
        except Exception:
            query_vec = None

        scored = []
        for memory in self.memories:
            if memory.importance < min_importance:
                continue

            # 向量相似度
            vector_score = 0.0
            if query_vec and memory.embedding:
                vector_score = cosine_similarity(query_vec, memory.embedding)

            # 关键词匹配
            keyword_score = self._keyword_score(query, memory.content)

            # 时间衰减
            time_decay = self._time_decay(memory.timestamp)

            # 重要性权重
            importance_weight = 0.8 + memory.importance * 0.4

            # 综合评分：(相似度 × 时间衰减) × 重要性权重
            base = vector_score * 0.7 + keyword_score * 0.3 if vector_score > 0 else keyword_score
            final_score = base * time_decay * importance_weight

            if final_score > 0:
                scored.append((final_score, memory))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def forget(self, strategy: str = "importance_based", **kwargs) -> int:
        """遗忘记忆"""
        if strategy == "importance_based":
            threshold = kwargs.get("threshold", 0.1)
            before = len(self.memories)
            self.memories = [m for m in self.memories if m.importance >= threshold]
            return before - len(self.memories)

        elif strategy == "time_based":
            max_age_days = kwargs.get("max_age_days", 30)
            cutoff = datetime.now() - timedelta(days=max_age_days)
            before = len(self.memories)
            self.memories = [
                m for m in self.memories
                if datetime.fromisoformat(m.timestamp) > cutoff
            ]
            return before - len(self.memories)

        elif strategy == "capacity_based":
            threshold = kwargs.get("threshold", 0.3)
            if len(self.memories) <= self.max_capacity:
                return 0
            excess = len(self.memories) - self.max_capacity
            sorted_memories = sorted(self.memories, key=lambda m: m.importance)
            to_remove = sorted_memories[:excess]
            for m in to_remove:
                self.memories.remove(m)
            return len(to_remove)

        return 0

    def clear(self) -> int:
        count = len(self.memories)
        self.memories.clear()
        return count

    def count(self) -> int:
        self._expire_old_memories()
        return len(self.memories)

    def get_stats(self) -> Dict[str, Any]:
        self._expire_old_memories()
        types = {}
        for m in self.memories:
            types[m.metadata.get("event_type", "general")] = \
                types.get(m.metadata.get("event_type", "general"), 0) + 1
        return {
            "total": len(self.memories),
            "capacity": self.max_capacity,
            "max_age_minutes": self.max_age_seconds // 60,
            "avg_importance": sum(m.importance for m in self.memories) / max(len(self.memories), 1),
            "by_type": types,
        }

    # ========== 内部方法 ==========

    def _expire_old_memories(self):
        """清理过期记忆"""
        now = time.time()
        self.memories = [
            m for m in self.memories
            if now - datetime.fromisoformat(m.timestamp).timestamp() < self.max_age_seconds
        ]

    def _remove_lowest_priority(self):
        """移除重要性最低的记忆"""
        if not self.memories:
            return
        min_idx = min(range(len(self.memories)), key=lambda i: self.memories[i].importance)
        self.memories.pop(min_idx)

    def _keyword_score(self, query: str, content: str) -> float:
        """关键词匹配得分"""
        query_lower = query.lower()
        content_lower = content.lower()

        # 整体包含
        if query_lower in content_lower:
            return 1.0

        # 逐词匹配
        query_words = set(query_lower.split())
        if not query_words:
            return 0.0

        matches = sum(1 for w in query_words if w in content_lower)
        return matches / len(query_words)

    def _time_decay(self, timestamp: str) -> float:
        """时间衰减因子（指数衰减）"""
        try:
            memory_time = datetime.fromisoformat(timestamp)
            age_hours = (datetime.now() - memory_time).total_seconds() / 3600
            decay = 0.1
            return max(0.1, math.exp(-decay * age_hours / 24))
        except Exception:
            return 0.5

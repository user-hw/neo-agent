"""记忆管理器 (MemoryManager)

统一协调和调度不同类型的记忆模块，提供统一的操作接口。
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import MemoryConfig, MemoryItem, BaseMemory
from .types.working import WorkingMemory
from .types.episodic import EpisodicMemory
from .types.semantic import SemanticMemory


class MemoryManager:
    """记忆管理器 - 统一的记忆操作接口

    管理 WorkingMemory / EpisodicMemory / SemanticMemory 三种记忆类型，
    提供 add / retrieve / forget / consolidate 等统一操作。
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        user_id: str = "default_user",
        enable_working: bool = True,
        enable_episodic: bool = True,
        enable_semantic: bool = True,
    ):
        self.config = config or MemoryConfig()
        self.user_id = user_id
        self.memory_types: Dict[str, BaseMemory] = {}

        if enable_working:
            self.memory_types["working"] = WorkingMemory(self.config)
        if enable_episodic:
            self.memory_types["episodic"] = EpisodicMemory(self.config)
        if enable_semantic:
            self.memory_types["semantic"] = SemanticMemory(self.config)

        enabled = list(self.memory_types.keys())
        print(f"🧠 MemoryManager 初始化完成，启用记忆类型: {enabled}")

    # ========== 添加 ==========

    def add_memory(
        self,
        content: str,
        memory_type: str = "working",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """添加一条记忆

        Args:
            content: 记忆内容
            memory_type: 记忆类型 (working / episodic / semantic)
            importance: 重要性 (0.0-1.0)
            metadata: 额外元数据

        Returns:
            记忆 ID
        """
        if memory_type not in self.memory_types:
            raise ValueError(f"不支持的记忆类型: {memory_type}")

        item = MemoryItem(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            content=content,
            memory_type=memory_type,
            importance=max(0.0, min(1.0, importance)),
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
        )

        return self.memory_types[memory_type].add(item)

    # ========== 检索 ==========

    def retrieve_memories(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[str]] = None,
        min_importance: float = 0.0,
        **kwargs,
    ) -> List[MemoryItem]:
        """跨类型检索记忆

        Args:
            query: 查询文本
            limit: 返回数量上限
            memory_types: 指定检索的记忆类型，None 表示全部
            min_importance: 最小重要性过滤

        Returns:
            匹配的记忆列表（按相关性排序）
        """
        types_to_search = memory_types or list(self.memory_types.keys())
        all_results: List[tuple] = []  # (score, item)

        for mtype in types_to_search:
            if mtype not in self.memory_types:
                continue
            try:
                results = self.memory_types[mtype].retrieve(
                    query, limit=limit * 2, min_importance=min_importance, **kwargs
                )
                for i, item in enumerate(results):
                    # 计算位置衰减得分
                    position_score = 1.0 - (i / (len(results) + 1)) * 0.3
                    all_results.append((position_score, item))
            except Exception as e:
                print(f"⚠️ 检索 {mtype} 记忆失败: {e}")

        # 按得分排序，去重
        seen_ids = set()
        unique_results = []
        for score, item in sorted(all_results, key=lambda x: x[0], reverse=True):
            if item.id not in seen_ids:
                seen_ids.add(item.id)
                unique_results.append(item)

        return unique_results[:limit]

    # ========== 遗忘 ==========

    def forget_memories(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30,
        **kwargs,
    ) -> int:
        """跨类型遗忘记忆

        Args:
            strategy: 遗忘策略 (importance_based / time_based / capacity_based)
            threshold: 重要性阈值
            max_age_days: 最大保留天数

        Returns:
            遗忘的总数量
        """
        total = 0
        for mtype, mem in self.memory_types.items():
            try:
                count = mem.forget(
                    strategy=strategy, threshold=threshold,
                    max_age_days=max_age_days, **kwargs
                )
                total += count
                if count > 0:
                    print(f"🧹 {mtype}: 遗忘 {count} 条记忆")
            except Exception as e:
                print(f"⚠️ {mtype} 遗忘失败: {e}")
        return total

    # ========== 整合 ==========

    def consolidate_memories(
        self,
        from_type: str = "working",
        to_type: str = "episodic",
        importance_threshold: float = 0.7,
    ) -> int:
        """将重要的短期记忆整合为长期记忆

        Args:
            from_type: 源记忆类型
            to_type: 目标记忆类型
            importance_threshold: 重要性阈值

        Returns:
            整合的记忆数量
        """
        if from_type not in self.memory_types or to_type not in self.memory_types:
            return 0

        source = self.memory_types[from_type]
        target = self.memory_types[to_type]

        # 获取高重要性的记忆
        results = source.retrieve("", limit=1000, min_importance=importance_threshold)

        count = 0
        for item in results:
            if item.importance >= importance_threshold:
                item.memory_type = to_type
                item.metadata["consolidated_from"] = from_type
                target.add(item)
                count += 1

        if count > 0:
            print(f"🔄 整合记忆: {count} 条 ({from_type} → {to_type})")
        return count

    # ========== 统计 ==========

    def get_summary(self, limit: int = 10) -> str:
        """获取记忆系统摘要"""
        lines = ["## 记忆系统摘要", ""]
        total = 0
        for mtype, mem in self.memory_types.items():
            c = mem.count()
            total += c
            lines.append(f"- **{mtype}**: {c} 条")
        lines.append(f"- **总计**: {total} 条")
        return "\n".join(lines)

    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有记忆模块的统计信息"""
        stats = {}
        for mtype, mem in self.memory_types.items():
            stats[mtype] = mem.get_stats()
        return stats

    def clear_all(self) -> int:
        """清空所有记忆"""
        total = 0
        for mem in self.memory_types.values():
            total += mem.clear()
        print(f"🧹 已清空所有记忆: {total} 条")
        return total

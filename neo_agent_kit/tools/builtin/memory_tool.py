"""记忆工具 (MemoryTool)

为 Agent 提供记忆能力的统一接口。

支持的操作:
- add: 添加记忆 (working/episodic/semantic)
- search: 搜索记忆
- summary: 获取记忆摘要
- stats: 获取统计信息
- forget: 遗忘记忆 (importance_based/time_based/capacity_based)
- consolidate: 整合记忆 (短期→长期)
- clear_all: 清空所有记忆
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..base import Tool, ToolParameter
from ...memory import MemoryManager, MemoryConfig


class MemoryTool(Tool):
    """记忆工具 - 为 Agent 提供记忆功能

    使用示例:
        memory_tool = MemoryTool(user_id="user123")
        memory_tool.run({"action": "add", "content": "用户叫张三", "memory_type": "semantic"})
        memory_tool.run({"action": "search", "query": "张三"})
    """

    def __init__(
        self,
        user_id: str = "default_user",
        memory_config: Optional[MemoryConfig] = None,
        memory_types: Optional[List[str]] = None,
    ):
        super().__init__(
            name="memory",
            description=(
                "记忆工具 - 可以存储和检索对话历史、知识和经验。"
                "支持操作: add(添加记忆), search(搜索记忆), summary(记忆摘要), "
                "stats(统计信息), forget(遗忘记忆), consolidate(整合记忆), clear_all(清空)"
            ),
        )

        self.memory_config = memory_config or MemoryConfig()
        self.memory_types = memory_types or ["working", "episodic", "semantic"]
        self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.memory_manager = MemoryManager(
            config=self.memory_config,
            user_id=user_id,
            enable_working="working" in self.memory_types,
            enable_episodic="episodic" in self.memory_types,
            enable_semantic="semantic" in self.memory_types,
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作类型: add, search, summary, stats, forget, consolidate, clear_all",
                required=True,
            ),
            ToolParameter(
                name="content",
                type="string",
                description="记忆内容 (用于 add 操作)",
                required=False,
            ),
            ToolParameter(
                name="query",
                type="string",
                description="搜索查询 (用于 search 操作)",
                required=False,
            ),
            ToolParameter(
                name="memory_type",
                type="string",
                description="记忆类型: working, episodic, semantic (默认 working)",
                required=False,
                default="working",
            ),
            ToolParameter(
                name="importance",
                type="string",
                description="重要性 0.0-1.0 (默认 0.5)",
                required=False,
                default="0.5",
            ),
            ToolParameter(
                name="strategy",
                type="string",
                description="遗忘策略: importance_based, time_based, capacity_based",
                required=False,
                default="importance_based",
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> str:
        """执行记忆操作"""
        action = parameters.get("action", "search")

        try:
            if action == "add":
                return self._add_memory(parameters)
            elif action == "search":
                return self._search_memory(parameters)
            elif action == "summary":
                return self.memory_manager.get_summary()
            elif action == "stats":
                return self._format_stats()
            elif action == "forget":
                return self._forget(parameters)
            elif action == "consolidate":
                return self._consolidate(parameters)
            elif action == "clear_all":
                count = self.memory_manager.clear_all()
                return f"🧹 已清空所有记忆 ({count} 条)"
            else:
                return f"❌ 不支持的操作: {action}。支持: add, search, summary, stats, forget, consolidate, clear_all"
        except Exception as e:
            return f"❌ 记忆操作失败: {str(e)}"

    # ========== 便捷方法 ==========

    def execute(self, action: str, **kwargs) -> str:
        """便捷执行方法（兼容 hello-agents 调用风格）"""
        kwargs["action"] = action
        return self.run(kwargs)

    def _add_memory(self, params: Dict[str, Any]) -> str:
        content = params.get("content", "")
        if not content:
            return "❌ 添加记忆失败: 内容不能为空"

        memory_type = params.get("memory_type", "working")
        importance = float(params.get("importance", 0.5))

        metadata = {
            "session_id": self.current_session_id,
            "timestamp": datetime.now().isoformat(),
        }

        # 传递额外的元数据
        for key in ["event_type", "concept", "knowledge_type"]:
            if key in params:
                metadata[key] = params[key]

        memory_id = self.memory_manager.add_memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata,
        )
        return f"✅ 记忆已添加 (ID: {memory_id[:12]}..., 类型: {memory_type})"

    def _search_memory(self, params: Dict[str, Any]) -> str:
        query = params.get("query", "")
        if not query:
            return "❌ 搜索失败: 查询不能为空"

        limit = int(params.get("limit", 5))
        memory_type = params.get("memory_type")
        min_importance = float(params.get("min_importance", 0.0))

        memory_types = [memory_type] if memory_type else None

        results = self.memory_manager.retrieve_memories(
            query=query,
            limit=limit,
            memory_types=memory_types,
            min_importance=min_importance,
        )

        if not results:
            return f"🔍 未找到与 '{query}' 相关的记忆"

        labels = {
            "working": "工作记忆",
            "episodic": "情景记忆",
            "semantic": "语义记忆",
        }

        lines = [f"🔍 找到 {len(results)} 条相关记忆:"]
        for i, mem in enumerate(results, 1):
            label = labels.get(mem.memory_type, mem.memory_type)
            preview = mem.content[:100] + "..." if len(mem.content) > 100 else mem.content
            lines.append(f"{i}. [{label}] {preview} (重要性: {mem.importance:.2f})")

        return "\n".join(lines)

    def _format_stats(self) -> str:
        stats = self.memory_manager.get_all_stats()
        lines = ["📊 记忆系统统计:"]
        for mtype, s in stats.items():
            lines.append(f"  - {mtype}: {s.get('total', 0)} 条")
        return "\n".join(lines)

    def _forget(self, params: Dict[str, Any]) -> str:
        strategy = params.get("strategy", "importance_based")
        threshold = float(params.get("threshold", 0.1))
        max_age_days = int(params.get("max_age_days", 30))
        count = self.memory_manager.forget_memories(
            strategy=strategy, threshold=threshold, max_age_days=max_age_days
        )
        return f"🧹 已遗忘 {count} 条记忆 (策略: {strategy})"

    def _consolidate(self, params: Dict[str, Any]) -> str:
        from_type = params.get("from_type", "working")
        to_type = params.get("to_type", "episodic")
        threshold = float(params.get("importance_threshold", 0.7))
        count = self.memory_manager.consolidate_memories(
            from_type=from_type, to_type=to_type, importance_threshold=threshold
        )
        return f"🔄 已整合 {count} 条记忆 ({from_type} → {to_type})"

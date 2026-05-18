"""上下文构建器 (ContextBuilder)

实现 GSSC (Gather-Select-Structure-Compress) 流水线，
提供统一的上下文管理接口。

核心理念：上下文是有限资源，需要像预算一样精心管理。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 核心数据结构
# ============================================================

@dataclass
class ContextPacket:
    """候选信息包 —— 上下文中的基本信息单元

    Attributes:
        content: 信息内容
        timestamp: 时间戳
        token_count: Token 数量
        relevance_score: 相关性分数 (0.0-1.0)
        metadata: 可选的元数据
    """

    content: str
    timestamp: datetime
    token_count: int
    relevance_score: float = 0.5
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.relevance_score = max(0.0, min(1.0, self.relevance_score))


@dataclass
class ContextConfig:
    """上下文构建配置

    Attributes:
        max_tokens: 最大 token 数量
        reserve_ratio: 为系统指令预留的比例 (0.0-1.0)
        min_relevance: 最低相关性阈值
        enable_compression: 是否启用压缩
        recency_weight: 新近性权重 (0.0-1.0)
        relevance_weight: 相关性权重 (0.0-1.0)
    """

    max_tokens: int = 3000
    reserve_ratio: float = 0.2
    min_relevance: float = 0.1
    enable_compression: bool = True
    recency_weight: float = 0.3
    relevance_weight: float = 0.7

    def __post_init__(self):
        assert 0.0 <= self.reserve_ratio <= 1.0, "reserve_ratio 必须在 [0, 1] 范围内"
        assert 0.0 <= self.min_relevance <= 1.0, "min_relevance 必须在 [0, 1] 范围内"
        assert abs(self.recency_weight + self.relevance_weight - 1.0) < 1e-6, \
            "recency_weight + relevance_weight 必须等于 1.0"


# ============================================================
# ContextBuilder 核心类
# ============================================================

class ContextBuilder:
    """上下文构建器 —— GSSC 流水线

    将 "获取(Gather)-选择(Select)-结构化(Structure)-压缩(Compress)"
    抽象为可复用流水线，统一管理 Agent 的上下文。

    使用示例:
        builder = ContextBuilder(memory_tool=mem, rag_tool=rag)
        context = builder.build(
            user_query="如何优化Pandas内存占用?",
            conversation_history=[...],
            system_instructions="你是一位数据工程顾问..."
        )
    """

    def __init__(
        self,
        memory_tool: Any = None,
        rag_tool: Any = None,
        config: Optional[ContextConfig] = None,
    ):
        self.memory_tool = memory_tool
        self.rag_tool = rag_tool
        self.config = config or ContextConfig()

    # ========== 主入口 ==========

    def build(
        self,
        user_query: str,
        conversation_history: Optional[List[Any]] = None,
        system_instructions: Optional[str] = None,
        custom_packets: Optional[List[ContextPacket]] = None,
    ) -> str:
        """构建优化的上下文

        Args:
            user_query: 用户查询
            conversation_history: 对话历史 (Message 列表)
            system_instructions: 系统指令
            custom_packets: 自定义信息包

        Returns:
            结构化的上下文字符串
        """
        # 1. Gather: 汇集候选信息
        packets = self._gather(
            user_query, conversation_history,
            system_instructions, custom_packets,
        )

        # 2. Select: 智能选择
        available = int(self.config.max_tokens * (1 - self.config.reserve_ratio))
        selected = self._select(packets, user_query, available)

        # 3. Structure: 结构化输出
        context = self._structure(selected, user_query)

        # 4. Compress: 兜底压缩
        if self.config.enable_compression:
            context = self._compress(context, self.config.max_tokens)

        return context

    # ========== Phase 1: Gather ==========

    def _gather(
        self,
        user_query: str,
        conversation_history: Optional[List[Any]] = None,
        system_instructions: Optional[str] = None,
        custom_packets: Optional[List[ContextPacket]] = None,
    ) -> List[ContextPacket]:
        """Gather: 多源信息汇集"""
        packets: List[ContextPacket] = []

        # 1. 系统指令 (最高优先级, 始终保留)
        if system_instructions:
            packets.append(ContextPacket(
                content=system_instructions,
                timestamp=datetime.now(),
                token_count=self._count_tokens(system_instructions),
                relevance_score=1.0,
                metadata={"type": "system_instruction", "priority": "high"},
            ))

        # 2. 从记忆系统检索
        if self.memory_tool:
            try:
                result = self.memory_tool.execute("search", query=user_query, limit=10)
                packets.extend(self._parse_memory_results(result, user_query))
            except Exception as e:
                print(f"[ContextBuilder] 记忆检索失败: {e}")

        # 3. 从 RAG 系统检索
        if self.rag_tool:
            try:
                result = self.rag_tool.execute("search", query=user_query, limit=5)
                packets.extend(self._parse_rag_results(result, user_query))
            except Exception as e:
                print(f"[ContextBuilder] RAG 检索失败: {e}")

        # 4. 对话历史 (仅保留最近5条)
        if conversation_history:
            for msg in conversation_history[-10:]:
                role = getattr(msg, 'role', 'user')
                content = getattr(msg, 'content', str(msg))
                ts = getattr(msg, 'timestamp', datetime.now())
                packets.append(ContextPacket(
                    content=f"{role}: {content}",
                    timestamp=ts if isinstance(ts, datetime) else datetime.now(),
                    token_count=self._count_tokens(content),
                    relevance_score=0.6,
                    metadata={"type": "conversation_history", "role": role},
                ))

        # 5. 自定义信息包
        if custom_packets:
            packets.extend(custom_packets)

        print(f"[ContextBuilder] 汇集了 {len(packets)} 个候选信息包")
        return packets

    def _parse_memory_results(self, result: str, query: str) -> List[ContextPacket]:
        """解析记忆检索结果为 ContextPacket"""
        packets = []
        lines = result.split("\n")
        for line in lines:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            # 提取内容
            if ". [" in line and "] " in line:
                content = line.split("] ", 1)[-1].split(" (重要性:")[0].strip()
            else:
                content = line
            packets.append(ContextPacket(
                content=f"[记忆] {content}",
                timestamp=datetime.now(),
                token_count=self._count_tokens(content),
                relevance_score=self._calculate_relevance(content, query),
                metadata={"type": "memory"},
            ))
        return packets

    def _parse_rag_results(self, result: str, query: str) -> List[ContextPacket]:
        """解析 RAG 检索结果为 ContextPacket"""
        packets = []
        lines = result.split("\n")
        current_content = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line[0].isdigit() and ". " in line:
                if current_content:
                    packets.append(ContextPacket(
                        content=f"[知识] {current_content}",
                        timestamp=datetime.now(),
                        token_count=self._count_tokens(current_content),
                        relevance_score=self._calculate_relevance(current_content, query),
                        metadata={"type": "rag_result"},
                    ))
                current_content = line.split(". ", 1)[-1] if ". " in line else line
            else:
                current_content += " " + line
        if current_content:
            packets.append(ContextPacket(
                content=f"[知识] {current_content}",
                timestamp=datetime.now(),
                token_count=self._count_tokens(current_content),
                relevance_score=self._calculate_relevance(current_content, query),
                metadata={"type": "rag_result"},
            ))
        return packets

    # ========== Phase 2: Select ==========

    def _select(
        self,
        packets: List[ContextPacket],
        user_query: str,
        available_tokens: int,
    ) -> List[ContextPacket]:
        """Select: 基于相关性和新近性的智能选择"""

        # 分离系统指令
        system_packets = [p for p in packets if p.metadata.get("type") == "system_instruction"]
        other_packets = [p for p in packets if p.metadata.get("type") != "system_instruction"]

        system_tokens = sum(p.token_count for p in system_packets)
        remaining = available_tokens - system_tokens

        if remaining <= 0:
            print("[ContextBuilder] ⚠️ 系统指令已占满 token 预算")
            return system_packets

        # 评分
        scored: List[Tuple[float, ContextPacket]] = []
        for packet in other_packets:
            if packet.relevance_score == 0.5:
                packet.relevance_score = self._calculate_relevance(
                    packet.content, user_query
                )
            recency = self._calculate_recency(packet.timestamp)
            combined = (
                self.config.relevance_weight * packet.relevance_score
                + self.config.recency_weight * recency
            )
            if packet.relevance_score >= self.config.min_relevance:
                scored.append((combined, packet))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 贪心填充
        selected = list(system_packets)
        current = system_tokens

        for _, packet in scored:
            if current + packet.token_count <= available_tokens:
                selected.append(packet)
                current += packet.token_count
            else:
                break

        print(f"[ContextBuilder] 选择了 {len(selected)} 个信息包, 共 {current} tokens")
        return selected

    def _calculate_relevance(self, content: str, query: str) -> float:
        """Jaccard 相似度计算相关性"""
        content_words = set(content.lower().split())
        query_words = set(query.lower().split())
        if not query_words:
            return 0.0
        intersection = content_words & query_words
        union = content_words | query_words
        return len(intersection) / len(union) if union else 0.0

    def _calculate_recency(self, timestamp: datetime) -> float:
        """指数衰减模型计算新近性"""
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        return max(0.1, min(1.0, math.exp(-0.1 * age_hours / 24)))

    # ========== Phase 3: Structure ==========

    def _structure(self, selected: List[ContextPacket], user_query: str) -> str:
        """Structure: 组织为分区模板"""
        system_instructions: List[str] = []
        evidence: List[str] = []
        context: List[str] = []

        for p in selected:
            ptype = p.metadata.get("type", "general")
            if ptype == "system_instruction":
                system_instructions.append(p.content)
            elif ptype in ("rag_result", "knowledge"):
                evidence.append(p.content)
            else:
                context.append(p.content)

        sections: List[str] = []

        if system_instructions:
            sections.append("[Role & Policies]\n" + "\n".join(system_instructions))

        sections.append(f"[Task]\n{user_query}")

        if evidence:
            sections.append("[Evidence]\n" + "\n---\n".join(evidence))

        if context:
            sections.append("[Context]\n" + "\n".join(context))

        sections.append("[Output]\n请基于以上信息，提供准确、有据的回答。")

        return "\n\n".join(sections)

    # ========== Phase 4: Compress ==========

    def _compress(self, context: str, max_tokens: int) -> str:
        """Compress: 兜底压缩超限上下文"""
        current = self._count_tokens(context)
        if current <= max_tokens:
            return context

        print(f"[ContextBuilder] 上下文超限 ({current} > {max_tokens}), 执行压缩")

        sections = context.split("\n\n")
        compressed: List[str] = []
        total = 0

        for section in sections:
            st = self._count_tokens(section)
            if total + st <= max_tokens:
                compressed.append(section)
                total += st
            else:
                remaining = max_tokens - total
                if remaining > 50:
                    compressed.append(
                        self._truncate_text(section, remaining)
                        + "\n[... 内容已压缩 ...]"
                    )
                break

        result = "\n\n".join(compressed)
        print(f"[ContextBuilder] 压缩完成: {current} -> {self._count_tokens(result)} tokens")
        return result

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """截断文本到指定 token 数"""
        char_per_token = max(len(text) / max(self._count_tokens(text), 1), 4)
        return text[: int(max_tokens * char_per_token)]

    # ========== 工具方法 ==========

    def _count_tokens(self, text: str) -> int:
        """估算 token 数量 (中文 1 char ≈ 1 token, 英文 1 word ≈ 1.3 token)"""
        chinese = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        english = len([w for w in text.split() if w])
        return int(chinese + english * 1.3)

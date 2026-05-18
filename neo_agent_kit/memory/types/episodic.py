"""情景记忆 (EpisodicMemory) 实现

特点:
- SQLite 持久化存储，支持结构化查询
- 按时间序列和会话组织
- 混合检索：语义相似度 + 时间近因性

模拟人类的情景记忆：记录具体的个人经历和事件。
"""
from __future__ import annotations

import os
import json
import math
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..base import BaseMemory, MemoryConfig, MemoryItem
from ..embedding import TFIDFEmbedding, cosine_similarity


class EpisodicMemory(BaseMemory):
    """情景记忆 - 存储具体事件和经历"""

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self._db_path = config.database_path
        self._tfidf = TFIDFEmbedding()
        self._sessions: Dict[str, List[str]] = {}  # session_id -> [memory_id, ...]
        self._init_db()

    def _init_db(self):
        """初始化 SQLite 数据库"""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                event_type TEXT,
                metadata TEXT,
                embedding TEXT
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodic_session
            ON episodic_memories(session_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodic_timestamp
            ON episodic_memories(timestamp)
        """)
        self._conn.commit()

        # 加载已有会话索引
        rows = self._conn.execute(
            "SELECT DISTINCT session_id FROM episodic_memories WHERE session_id IS NOT NULL"
        ).fetchall()
        for (sid,) in rows:
            mem_ids = [r[0] for r in self._conn.execute(
                "SELECT id FROM episodic_memories WHERE session_id=? ORDER BY timestamp", (sid,)
            ).fetchall()]
            self._sessions[sid] = mem_ids

    # ========== 核心操作 ==========

    def add(self, memory_item: MemoryItem) -> str:
        """添加情景记忆"""
        memory_item.memory_type = "episodic"
        if not memory_item.id:
            memory_item.id = f"ep_{uuid.uuid4().hex[:12]}"

        session_id = memory_item.metadata.get("session_id", "default")
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(memory_item.id)

        # 生成嵌入
        try:
            emb = self._tfidf.encode(memory_item.content)[0]
            memory_item.embedding = emb
            emb_json = json.dumps(emb)
        except Exception:
            emb_json = "[]"

        self._conn.execute(
            """INSERT OR REPLACE INTO episodic_memories
               (id, content, importance, timestamp, session_id, event_type, metadata, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory_item.id,
                memory_item.content,
                memory_item.importance,
                memory_item.timestamp,
                session_id,
                memory_item.metadata.get("event_type", "general"),
                json.dumps(memory_item.metadata, ensure_ascii=False),
                emb_json,
            ),
        )
        self._conn.commit()
        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """混合检索：结构化过滤 + 语义匹配"""
        session_id = kwargs.get("session_id")
        event_type = kwargs.get("event_type")
        min_importance = kwargs.get("min_importance", 0.0)

        # 构建 SQL 过滤
        where = ["1=1"]
        params = []
        if session_id:
            where.append("session_id = ?")
            params.append(session_id)
        if event_type:
            where.append("event_type = ?")
            params.append(event_type)
        if min_importance > 0:
            where.append("importance >= ?")
            params.append(min_importance)

        rows = self._conn.execute(
            f"SELECT * FROM episodic_memories WHERE {' AND '.join(where)} ORDER BY timestamp DESC",
            params,
        ).fetchall()

        if not rows:
            return []

        # 语义匹配 + 评分
        try:
            query_vec = self._tfidf.encode(query)[0]
        except Exception:
            query_vec = None

        scored = []
        for row in rows:
            mid, content, importance, ts, sid, etype, meta_str, emb_str = row

            # 语义相似度
            vector_score = 0.0
            if query_vec and emb_str and emb_str != "[]":
                try:
                    emb = json.loads(emb_str)
                    vector_score = cosine_similarity(query_vec, emb)
                except Exception:
                    pass

            # 关键词匹配
            keyword_score = self._keyword_score(query, content)

            # 时间近因性
            recency = self._recency_score(ts)

            # 综合评分
            base = vector_score * 0.8 + max(keyword_score, vector_score) * 0.2
            if base == 0:
                base = keyword_score
            importance_weight = 0.8 + importance * 0.4
            final_score = base * recency * importance_weight

            if final_score > 0:
                try:
                    metadata = json.loads(meta_str)
                except Exception:
                    metadata = {}
                item = MemoryItem(
                    id=mid, content=content, memory_type="episodic",
                    importance=importance, timestamp=ts, metadata=metadata,
                )
                scored.append((final_score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def forget(self, strategy: str = "importance_based", **kwargs) -> int:
        """遗忘记忆"""
        if strategy == "importance_based":
            threshold = kwargs.get("threshold", 0.1)
            cursor = self._conn.execute(
                "DELETE FROM episodic_memories WHERE importance < ?", (threshold,)
            )
            self._conn.commit()
            return cursor.rowcount

        elif strategy == "time_based":
            max_age_days = kwargs.get("max_age_days", 30)
            cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
            cursor = self._conn.execute(
                "DELETE FROM episodic_memories WHERE timestamp < ?", (cutoff,)
            )
            self._conn.commit()
            return cursor.rowcount

        elif strategy == "capacity_based":
            threshold = kwargs.get("threshold", 0.3)
            max_cap = kwargs.get("max_capacity", 1000)
            count = self._conn.execute("SELECT COUNT(*) FROM episodic_memories").fetchone()[0]
            if count <= max_cap:
                return 0
            excess = count - max_cap
            cursor = self._conn.execute(
                "DELETE FROM episodic_memories WHERE id IN "
                "(SELECT id FROM episodic_memories ORDER BY importance ASC LIMIT ?)",
                (excess,),
            )
            self._conn.commit()
            return cursor.rowcount

        return 0

    def clear(self) -> int:
        count = self._conn.execute("SELECT COUNT(*) FROM episodic_memories").fetchone()[0]
        self._conn.execute("DELETE FROM episodic_memories")
        self._conn.commit()
        self._sessions.clear()
        return count

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM episodic_memories").fetchone()[0]

    def get_stats(self) -> Dict[str, Any]:
        total = self.count()
        sessions = len(self._sessions)
        avg_imp = self._conn.execute(
            "SELECT AVG(importance) FROM episodic_memories"
        ).fetchone()[0] or 0.0
        event_counts = {}
        for (etype, cnt) in self._conn.execute(
            "SELECT event_type, COUNT(*) FROM episodic_memories GROUP BY event_type"
        ).fetchall():
            event_counts[etype or "general"] = cnt

        return {
            "total": total,
            "sessions": sessions,
            "avg_importance": round(avg_imp, 3),
            "by_event_type": event_counts,
        }

    # ========== 会话方法 ==========

    def get_session_ids(self) -> List[str]:
        return list(self._sessions.keys())

    def get_session_memories(self, session_id: str) -> List[MemoryItem]:
        rows = self._conn.execute(
            "SELECT * FROM episodic_memories WHERE session_id=? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
        items = []
        for row in rows:
            mid, content, importance, ts, sid, etype, meta_str, emb_str = row
            try:
                metadata = json.loads(meta_str)
            except Exception:
                metadata = {}
            items.append(MemoryItem(
                id=mid, content=content, memory_type="episodic",
                importance=importance, timestamp=ts, metadata=metadata,
            ))
        return items

    # ========== 内部方法 ==========

    def _keyword_score(self, query: str, content: str) -> float:
        query_lower = query.lower()
        content_lower = content.lower()
        if query_lower in content_lower:
            return 1.0
        words = set(query_lower.split())
        if not words:
            return 0.0
        return sum(1 for w in words if w in content_lower) / len(words)

    def _recency_score(self, timestamp: str) -> float:
        try:
            t = datetime.fromisoformat(timestamp)
            hours = (datetime.now() - t).total_seconds() / 3600
            return max(0.1, math.exp(-0.1 * hours / 24))
        except Exception:
            return 0.5

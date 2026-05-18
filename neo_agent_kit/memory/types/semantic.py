"""语义记忆 (SemanticMemory) 实现

特点:
- 存储抽象概念、知识和规则
- 支持关键词检索 + 加权评分
- 可扩展为知识图谱（可选安装 networkx）

模拟人类的语义记忆：存储"巴黎是法国首都"这类一般知识。
"""
from __future__ import annotations

import json
import math
import sqlite3
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..base import BaseMemory, MemoryConfig, MemoryItem
from ..embedding import TFIDFEmbedding, cosine_similarity


class SemanticMemory(BaseMemory):
    """语义记忆 - 抽象知识和概念的存储"""

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self._db_path = config.database_path
        self._tfidf = TFIDFEmbedding()
        self._entities: Dict[str, Dict[str, Any]] = {}
        self._relations: List[Dict[str, str]] = []
        self._init_db()

    def _init_db(self):
        """初始化语义记忆数据库"""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                timestamp TEXT NOT NULL,
                knowledge_type TEXT,
                concept TEXT,
                metadata TEXT,
                embedding TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT,
                properties TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_relations (
                id TEXT PRIMARY KEY,
                source_id TEXT,
                target_id TEXT,
                relation_type TEXT,
                weight REAL DEFAULT 1.0
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_semantic_concept ON semantic_memories(concept)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_semantic_type ON semantic_memories(knowledge_type)")
        self._conn.commit()

        # 加载已有实体
        for row in self._conn.execute("SELECT * FROM semantic_entities").fetchall():
            eid, name, etype, props = row
            try:
                props_dict = json.loads(props) if props else {}
            except Exception:
                props_dict = {}
            self._entities[eid] = {"name": name, "type": etype, "properties": props_dict}

    # ========== 核心操作 ==========

    def add(self, memory_item: MemoryItem) -> str:
        """添加语义记忆，同时提取实体和关系"""
        memory_item.memory_type = "semantic"
        if not memory_item.id:
            memory_item.id = f"sm_{uuid.uuid4().hex[:12]}"

        concept = memory_item.metadata.get("concept", "general")
        knowledge_type = memory_item.metadata.get("knowledge_type", "factual")

        # 生成嵌入
        try:
            emb = self._tfidf.encode(memory_item.content)[0]
            memory_item.embedding = emb
            emb_json = json.dumps(emb)
        except Exception:
            emb_json = "[]"

        self._conn.execute(
            """INSERT OR REPLACE INTO semantic_memories
               (id, content, importance, timestamp, knowledge_type, concept, metadata, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory_item.id, memory_item.content, memory_item.importance,
                memory_item.timestamp, knowledge_type, concept,
                json.dumps(memory_item.metadata, ensure_ascii=False), emb_json,
            ),
        )

        # 提取实体
        entities = self._extract_entities(memory_item.content)
        for entity_name in entities:
            entity_id = f"ent_{uuid.uuid4().hex[:8]}"
            self._entities[entity_id] = {"name": entity_name, "type": "auto", "properties": {}}
            self._conn.execute(
                "INSERT OR IGNORE INTO semantic_entities (id, name, entity_type, properties) VALUES (?, ?, ?, ?)",
                (entity_id, entity_name, "auto", "{}"),
            )

        self._conn.commit()
        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索语义记忆（向量 + 关键词 + 概念匹配）"""
        concept = kwargs.get("concept")
        knowledge_type = kwargs.get("knowledge_type")
        min_importance = kwargs.get("min_importance", 0.0)

        where = ["1=1"]
        params = []
        if concept:
            where.append("concept = ?")
            params.append(concept)
        if knowledge_type:
            where.append("knowledge_type = ?")
            params.append(knowledge_type)
        if min_importance > 0:
            where.append("importance >= ?")
            params.append(min_importance)

        rows = self._conn.execute(
            f"SELECT * FROM semantic_memories WHERE {' AND '.join(where)}", params
        ).fetchall()

        if not rows:
            return []

        try:
            query_vec = self._tfidf.encode(query)[0]
        except Exception:
            query_vec = None

        scored = []
        for row in rows:
            mid, content, importance, ts, ktype, concept_val, meta_str, emb_str = row

            vector_score = 0.0
            if query_vec and emb_str and emb_str != "[]":
                try:
                    emb = json.loads(emb_str)
                    vector_score = cosine_similarity(query_vec, emb)
                except Exception:
                    pass

            keyword_score = self._keyword_score(query, content)
            entity_score = self._entity_match_score(query)

            # 混合评分：向量 50% + 关键词 30% + 实体 20%
            base = vector_score * 0.5 + keyword_score * 0.3 + entity_score * 0.2
            if base == 0:
                base = keyword_score
            importance_weight = 0.8 + importance * 0.4
            final_score = base * importance_weight

            if final_score > 0:
                try:
                    metadata = json.loads(meta_str)
                except Exception:
                    metadata = {}
                item = MemoryItem(
                    id=mid, content=content, memory_type="semantic",
                    importance=importance, timestamp=ts, metadata=metadata,
                )
                scored.append((final_score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def forget(self, strategy: str = "importance_based", **kwargs) -> int:
        if strategy == "importance_based":
            threshold = kwargs.get("threshold", 0.1)
            cursor = self._conn.execute(
                "DELETE FROM semantic_memories WHERE importance < ?", (threshold,)
            )
            self._conn.commit()
            return cursor.rowcount

        elif strategy == "time_based":
            max_age_days = kwargs.get("max_age_days", 30)
            cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
            cursor = self._conn.execute(
                "DELETE FROM semantic_memories WHERE timestamp < ?", (cutoff,)
            )
            self._conn.commit()
            return cursor.rowcount

        return 0

    def clear(self) -> int:
        count = self._conn.execute("SELECT COUNT(*) FROM semantic_memories").fetchone()[0]
        self._conn.execute("DELETE FROM semantic_memories")
        self._conn.execute("DELETE FROM semantic_entities")
        self._conn.execute("DELETE FROM semantic_relations")
        self._conn.commit()
        self._entities.clear()
        self._relations.clear()
        return count

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM semantic_memories").fetchone()[0]

    def get_stats(self) -> Dict[str, Any]:
        total = self.count()
        entities = len(self._entities)
        relations = len(self._relations)
        avg_imp = self._conn.execute(
            "SELECT AVG(importance) FROM semantic_memories"
        ).fetchone()[0] or 0.0

        type_counts = {}
        for (ktype, cnt) in self._conn.execute(
            "SELECT knowledge_type, COUNT(*) FROM semantic_memories GROUP BY knowledge_type"
        ).fetchall():
            type_counts[ktype or "factual"] = cnt

        return {
            "total": total,
            "entities": entities,
            "relations": relations,
            "avg_importance": round(avg_imp, 3),
            "by_type": type_counts,
        }

    # ========== 知识图谱方法 ==========

    def add_entity(self, name: str, entity_type: str = "concept", properties: Dict = None) -> str:
        entity_id = f"ent_{uuid.uuid4().hex[:8]}"
        self._entities[entity_id] = {
            "name": name, "type": entity_type, "properties": properties or {},
        }
        self._conn.execute(
            "INSERT OR REPLACE INTO semantic_entities (id, name, entity_type, properties) VALUES (?, ?, ?, ?)",
            (entity_id, name, entity_type, json.dumps(properties or {}, ensure_ascii=False)),
        )
        self._conn.commit()
        return entity_id

    def add_relation(self, source: str, target: str, relation_type: str, weight: float = 1.0) -> str:
        rel_id = f"rel_{uuid.uuid4().hex[:8]}"
        self._relations.append({
            "id": rel_id, "source": source, "target": target,
            "type": relation_type, "weight": weight,
        })
        self._conn.execute(
            "INSERT INTO semantic_relations (id, source_id, target_id, relation_type, weight) VALUES (?, ?, ?, ?, ?)",
            (rel_id, source, target, relation_type, weight),
        )
        self._conn.commit()
        return rel_id

    def get_entities(self) -> List[Dict[str, Any]]:
        return [{"id": k, **v} for k, v in self._entities.items()]

    # ========== 内部方法 ==========

    def _extract_entities(self, text: str) -> List[str]:
        """简单实体提取（基于大写词和专有名词模式）"""
        import re
        # 提取英文大写词短语、中文双字以上词
        entities = []
        # 英文专有名词
        for match in re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text):
            entities.append(match.group())
        # 中文双字及以上组合
        for match in re.finditer(r'[\u4e00-\u9fff]{2,}', text):
            entities.append(match.group())
        return list(set(entities))[:20]

    def _keyword_score(self, query: str, content: str) -> float:
        query_lower = query.lower()
        content_lower = content.lower()
        if query_lower in content_lower:
            return 1.0
        words = set(query_lower.split())
        if not words:
            return 0.0
        return sum(1 for w in words if w in content_lower) / len(words)

    def _entity_match_score(self, query: str) -> float:
        """查询与已知实体的匹配度"""
        if not self._entities:
            return 0.0
        query_lower = query.lower()
        matches = sum(
            1 for e in self._entities.values()
            if e["name"].lower() in query_lower
        )
        return min(1.0, matches / max(len(self._entities), 1) * 10)

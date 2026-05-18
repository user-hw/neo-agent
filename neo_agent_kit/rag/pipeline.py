"""RAG 管道模块

提供端到端的 RAG (检索增强生成) 管道:
- 文档索引: 分块 → 向量化 → 存储
- 查询检索: 向量检索 + 关键词检索
- 高级检索: MQE (多查询扩展) + HyDE (假设文档嵌入)
- LLM 增强生成
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .document import Document, DocumentProcessor
from ..memory.embedding import TFIDFEmbedding, cosine_similarity


class RAGPipeline:
    """RAG 处理管道"""

    def __init__(
        self,
        knowledge_base_path: str = "./knowledge_base",
        collection_name: str = "rag_knowledge_base",
        rag_namespace: str = "default",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.knowledge_base_path = knowledge_base_path
        self.collection_name = collection_name
        self.rag_namespace = rag_namespace
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.doc_processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.embedder = TFIDFEmbedding()

        # 初始化存储
        os.makedirs(knowledge_base_path, exist_ok=True)
        self._db_path = os.path.join(knowledge_base_path, f"rag_{rag_namespace}.db")
        self._init_db()

        print(f"📚 RAG 管道初始化: namespace={rag_namespace}, chunks={self.count_chunks()}")

    def _init_db(self):
        """初始化 SQLite 存储"""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                heading_path TEXT,
                embedding TEXT,
                doc_id TEXT,
                doc_source TEXT,
                namespace TEXT,
                char_count INTEGER,
                created_at TEXT
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_rag_namespace ON rag_chunks(namespace)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_rag_doc ON rag_chunks(doc_id)")
        self._conn.commit()

    # ========== 文档索引 ==========

    def index_document(self, document: Document) -> int:
        """索引文档: 分块 → 向量化 → 存储"""
        chunks = self.doc_processor.split_document(document)
        print(f"📄 文档分块: {len(chunks)} 个片段 (文档长度: {len(document.content)} 字符)")

        count = 0
        for chunk in chunks:
            chunk_id = f"chunk_{uuid.uuid4().hex[:12]}"
            content = chunk["content"]

            # 向量化
            try:
                vec = self.embedder.encode(content)[0]
            except Exception:
                vec = [0.0] * self.embedder.dimension

            self._conn.execute(
                """INSERT OR REPLACE INTO rag_chunks
                   (id, content, heading_path, embedding, doc_id, doc_source, namespace, char_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    chunk_id,
                    content,
                    chunk.get("heading_path"),
                    json.dumps(vec),
                    document.doc_id,
                    document.metadata.get("source", ""),
                    self.rag_namespace,
                    chunk.get("char_count", len(content)),
                    datetime.now().isoformat(),
                ),
            )
            count += 1

        self._conn.commit()
        print(f"✅ 索引完成: {count} 个片段已存储")
        return count

    def index_text(
        self, text: str, doc_id: Optional[str] = None, metadata: Optional[Dict] = None
    ) -> int:
        """索引文本字符串"""
        doc = self.doc_processor.load_text(text, doc_id, metadata)
        return self.index_document(doc)

    def index_file(self, file_path: str) -> int:
        """索引文件"""
        doc = self.doc_processor.load_file(file_path)
        if doc is None:
            return 0
        return self.index_document(doc)

    # ========== 查询检索 ==========

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        enable_mqe: bool = False,
        enable_hyde: bool = False,
    ) -> List[Dict[str, Any]]:
        """检索相关文档片段

        Args:
            query: 查询文本
            top_k: 返回数量
            min_score: 最低相似度阈值
            enable_mqe: 是否启用多查询扩展
            enable_hyde: 是否启用假设文档嵌入

        Returns:
            相关片段列表
        """
        # 查询扩展
        queries = [query]

        if enable_mqe:
            expansions = self._expand_query_mqe(query)
            queries.extend(expansions)

        if enable_hyde:
            hyde_doc = self._generate_hyde(query)
            if hyde_doc:
                queries.append(hyde_doc)

        # 收集所有扩展查询的结果
        all_hits = {}
        for q in queries:
            hits = self._vector_search(q, top_k * 3, min_score)
            for h in hits:
                chunk_id = h["id"]
                if chunk_id not in all_hits or h["score"] > all_hits[chunk_id].get("score", 0):
                    all_hits[chunk_id] = h

        # 排序返回
        sorted_hits = sorted(all_hits.values(), key=lambda x: x["score"], reverse=True)
        return sorted_hits[:top_k]

    def _vector_search(self, query: str, limit: int, min_score: float) -> List[Dict[str, Any]]:
        """向量相似度搜索"""
        try:
            query_vec = self.embedder.encode(query)[0]
        except Exception:
            return []

        rows = self._conn.execute(
            "SELECT id, content, heading_path, embedding, doc_source, char_count FROM rag_chunks WHERE namespace=?",
            (self.rag_namespace,),
        ).fetchall()

        scored = []
        for row in rows:
            chunk_id, content, heading, emb_str, doc_source, char_count = row
            try:
                emb = json.loads(emb_str)
            except Exception:
                continue

            score = cosine_similarity(query_vec, emb)
            if score < min_score:
                continue

            scored.append({
                "id": chunk_id,
                "content": content,
                "heading_path": heading,
                "score": score,
                "doc_source": doc_source,
                "char_count": char_count,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    # ========== 高级检索 ==========

    def _expand_query_mqe(self, query: str, n: int = 3) -> List[str]:
        """多查询扩展 - 生成同义查询"""
        # 使用基于规则的方法生成变体
        expansions = []

        # 变体1: 去掉问号
        if "?" in query or "？" in query:
            expansions.append(query.replace("?", "").replace("？", ""))

        # 变体2: 提取关键词
        words = query.replace("?", "").replace("？", "").split()
        if len(words) > 3:
            expansions.append(" ".join(words[:len(words)//2]))

        # 变体3-4: 添加/移除 "什么是" "如何" 等前缀
        prefixes = ["什么是", "如何", "请解释", "介绍一下"]
        for prefix in prefixes:
            if not query.startswith(prefix):
                expansions.append(f"{prefix}{query}")

        return list(set(expansions))[:n]

    def _generate_hyde(self, query: str) -> Optional[str]:
        """生成假设文档 (HyDE) - 用答案找答案

        使用 LLM 生成一个假设的答案段落，用于改善检索精度。
        如果没有 LLM，返回 None。
        """
        return None  # 需要 LLM 调用，由上层 Tool 处理

    # ========== 管理方法 ==========

    def count_chunks(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM rag_chunks WHERE namespace=?", (self.rag_namespace,)
        ).fetchone()[0]

    def get_stats(self) -> Dict[str, Any]:
        total = self.count_chunks()
        doc_count = self._conn.execute(
            "SELECT COUNT(DISTINCT doc_id) FROM rag_chunks WHERE namespace=?",
            (self.rag_namespace,),
        ).fetchone()[0]

        return {
            "namespace": self.rag_namespace,
            "collection": self.collection_name,
            "total_chunks": total,
            "total_documents": doc_count,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    def delete_document(self, doc_id: str) -> int:
        cursor = self._conn.execute(
            "DELETE FROM rag_chunks WHERE doc_id=? AND namespace=?",
            (doc_id, self.rag_namespace),
        )
        self._conn.commit()
        return cursor.rowcount

    def clear_namespace(self) -> int:
        cursor = self._conn.execute(
            "DELETE FROM rag_chunks WHERE namespace=?", (self.rag_namespace,)
        )
        self._conn.commit()
        return cursor.rowcount


def create_rag_pipeline(
    knowledge_base_path: str = "./knowledge_base",
    collection_name: str = "rag_knowledge_base",
    rag_namespace: str = "default",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> RAGPipeline:
    """创建 RAG 管道的工厂函数"""
    return RAGPipeline(
        knowledge_base_path=knowledge_base_path,
        collection_name=collection_name,
        rag_namespace=rag_namespace,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

"""RAG 工具 (RAGTool)

为 Agent 提供知识检索增强生成能力。

支持的操作:
- add_text: 添加文本到知识库
- add_document: 添加文件到知识库
- search: 搜索知识库
- ask: 基于知识库的智能问答 (需要 LLM)
- stats: 获取知识库统计
- delete: 删除文档
- clear: 清空命名空间
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..base import Tool, ToolParameter
from ...rag import RAGPipeline, create_rag_pipeline


class RAGTool(Tool):
    """RAG 工具 - 知识检索增强生成

    使用示例:
        rag_tool = RAGTool(knowledge_base_path="./knowledge_base")
        rag_tool.run({"action": "add_text", "text": "Python 是一种编程语言...", "document_id": "doc1"})
        rag_tool.run({"action": "search", "query": "什么是Python", "limit": "3"})
        rag_tool.run({"action": "ask", "question": "Python 的创始人是谁？"})
    """

    def __init__(
        self,
        knowledge_base_path: str = "./knowledge_base",
        collection_name: str = "rag_knowledge_base",
        rag_namespace: str = "default",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        super().__init__(
            name="rag",
            description=(
                "RAG (检索增强生成) 工具 - 支持向知识库添加文档/文本、检索相关内容和智能问答。"
                "支持操作: add_text(添加文本), add_document(添加文件), "
                "search(搜索), ask(智能问答), stats(统计), delete(删除), clear(清空)"
            ),
        )

        self.knowledge_base_path = knowledge_base_path
        self.collection_name = collection_name
        self.rag_namespace = rag_namespace

        self.pipeline = create_rag_pipeline(
            knowledge_base_path=knowledge_base_path,
            collection_name=collection_name,
            rag_namespace=rag_namespace,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        print(f"✅ RAG 工具初始化成功: namespace={rag_namespace}, collection={collection_name}")

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作类型: add_text, add_document, search, ask, stats, delete, clear",
                required=True,
            ),
            ToolParameter(
                name="text",
                type="string",
                description="要添加的文本内容 (用于 add_text)",
                required=False,
            ),
            ToolParameter(
                name="file_path",
                type="string",
                description="要添加的文件路径 (用于 add_document)",
                required=False,
            ),
            ToolParameter(
                name="query",
                type="string",
                description="搜索查询 (用于 search)",
                required=False,
            ),
            ToolParameter(
                name="question",
                type="string",
                description="要提问的问题 (用于 ask)",
                required=False,
            ),
            ToolParameter(
                name="limit",
                type="string",
                description="返回结果数量 (默认 5)",
                required=False,
                default="5",
            ),
            ToolParameter(
                name="document_id",
                type="string",
                description="文档 ID (用于 add_text 和 delete)",
                required=False,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> str:
        """执行 RAG 操作"""
        action = parameters.get("action", "search")

        try:
            if action == "add_text":
                return self._add_text(parameters)
            elif action == "add_document":
                return self._add_document(parameters)
            elif action == "search":
                return self._search(parameters)
            elif action == "ask":
                return self._ask(parameters)
            elif action == "stats":
                return self._stats()
            elif action == "delete":
                return self._delete(parameters)
            elif action == "clear":
                return self._clear()
            else:
                return f"❌ 不支持的操作: {action}。支持: add_text, add_document, search, ask, stats, delete, clear"
        except Exception as e:
            return f"❌ RAG 操作失败: {str(e)}"

    # ========== 便捷方法 ==========

    def execute(self, action: str, **kwargs) -> str:
        """便捷执行方法（兼容 hello-agents 调用风格）"""
        kwargs["action"] = action
        return self.run(kwargs)

    def _add_text(self, params: Dict[str, Any]) -> str:
        text = params.get("text", "")
        if not text:
            return "❌ 添加失败: 文本内容不能为空"

        doc_id = params.get("document_id")
        count = self.pipeline.index_text(text, doc_id=doc_id)
        return f"✅ 已添加文本到知识库 ({count} 个片段)"

    def _add_document(self, params: Dict[str, Any]) -> str:
        file_path = params.get("file_path", "")
        if not file_path:
            return "❌ 添加失败: 文件路径不能为空"

        if not os.path.exists(file_path):
            return f"❌ 文件不存在: {file_path}"

        count = self.pipeline.index_file(file_path)
        if count == 0:
            return f"❌ 文件处理失败: {file_path}"
        return f"✅ 已添加文档 '{os.path.basename(file_path)}' ({count} 个片段)"

    def _search(self, params: Dict[str, Any]) -> str:
        query = params.get("query", "")
        if not query:
            return "❌ 搜索失败: 查询不能为空"

        limit = int(params.get("limit", 5))
        min_score = float(params.get("min_score", 0.0))
        enable_mqe = params.get("enable_mqe", "false").lower() == "true"
        enable_hyde = params.get("enable_hyde", "false").lower() == "true"

        results = self.pipeline.search(
            query=query,
            top_k=limit,
            min_score=min_score,
            enable_mqe=enable_mqe,
            enable_hyde=enable_hyde,
        )

        if not results:
            return f"🔍 未找到与 '{query}' 相关的内容"

        lines = [f"🔍 找到 {len(results)} 条相关内容:"]
        for i, r in enumerate(results, 1):
            heading = f" [{r.get('heading_path', '')}]" if r.get("heading_path") else ""
            preview = r["content"][:150] + "..." if len(r["content"]) > 150 else r["content"]
            lines.append(
                f"{i}.{heading} (相似度: {r['score']:.3f})\n   {preview}"
            )

        return "\n".join(lines)

    def _ask(self, params: Dict[str, Any]) -> str:
        """基于知识库的智能问答

        先检索相关片段，再构建上下文提示。
        注意：需要外部 LLM 来完成最终生成。
        """
        question = params.get("question", "")
        if not question:
            return "❌ 提问失败: 问题不能为空"

        limit = int(params.get("limit", 5))

        # 检索相关片段
        results = self.pipeline.search(query=question, top_k=limit)

        if not results:
            return f"🔍 知识库中未找到与 '{question}' 相关的信息"

        # 构建上下文
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(f"[来源{i}] {r['content']}")

        context = "\n\n".join(context_parts)

        # 返回上下文（由 Agent 的 LLM 进行最终生成）
        return (
            f"📚 从知识库检索到 {len(results)} 条相关信息:\n\n"
            f"{context}\n\n"
            f"---\n"
            f"请基于以上信息回答问题: {question}"
        )

    def _stats(self) -> str:
        s = self.pipeline.get_stats()
        return (
            f"📊 知识库统计:\n"
            f"  - 命名空间: {s['namespace']}\n"
            f"  - 文档数: {s['total_documents']}\n"
            f"  - 片段数: {s['total_chunks']}\n"
            f"  - 分块大小: {s['chunk_size']}\n"
            f"  - 重叠大小: {s['chunk_overlap']}"
        )

    def _delete(self, params: Dict[str, Any]) -> str:
        doc_id = params.get("document_id", "")
        if not doc_id:
            return "❌ 删除失败: 需要指定 document_id"
        count = self.pipeline.delete_document(doc_id)
        return f"🗑️ 已删除文档 '{doc_id}' ({count} 个片段)"

    def _clear(self) -> str:
        count = self.pipeline.clear_namespace()
        return f"🧹 已清空命名空间 '{self.rag_namespace}' ({count} 个片段)"

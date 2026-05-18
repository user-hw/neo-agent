"""结构化笔记工具 (NoteTool)

为智能体提供持久化笔记管理，适用于长时程任务追踪。

格式: Markdown + YAML 前置元数据
- 机器可解析 (YAML 元数据)
- 人类可读 (Markdown 正文)
- Git 版本控制友好

支持7种操作: create / read / update / search / list / summary / delete
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..base import Tool, ToolParameter

# 尝试导入 yaml，失败则使用简单解析
try:
    import yaml
    _has_yaml = True
except ImportError:
    _has_yaml = False


class NoteTool(Tool):
    """结构化笔记工具

    使用 Markdown + YAML 格式存储笔记，支持:
    - create: 创建笔记
    - read: 读取笔记
    - update: 更新笔记
    - search: 搜索笔记 (标题+内容+标签)
    - list: 列出笔记 (可按类型/标签过滤)
    - summary: 笔记摘要统计
    - delete: 删除笔记

    笔记类型: task_state / conclusion / blocker / action / reference / general
    """

    def __init__(self, workspace: str = "./notes"):
        super().__init__(
            name="note",
            description=(
                "结构化笔记工具 - 支持创建/读取/搜索/列出笔记。"
                "笔记类型: task_state(任务状态), conclusion(结论), "
                "blocker(阻塞), action(行动), reference(参考), general(通用)。"
            ),
        )
        self.workspace = os.path.abspath(workspace)
        os.makedirs(self.workspace, exist_ok=True)

        # 索引文件
        self._index_path = os.path.join(self.workspace, "notes_index.json")
        self.index: Dict[str, Dict[str, Any]] = self._load_index()

        print(f"📝 NoteTool 初始化: workspace={self.workspace}, notes={len(self.index)}")

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="action", type="string",
                          description="操作: create, read, update, search, list, summary, delete",
                          required=True),
            ToolParameter(name="title", type="string",
                          description="笔记标题 (用于 create/update)", required=False),
            ToolParameter(name="content", type="string",
                          description="笔记内容, Markdown格式 (用于 create/update)", required=False),
            ToolParameter(name="note_type", type="string",
                          description="笔记类型: task_state/conclusion/blocker/action/reference/general",
                          required=False, default="general"),
            ToolParameter(name="tags", type="string",
                          description="标签列表, 逗号分隔 (如: refactoring,phase1)", required=False),
            ToolParameter(name="query", type="string",
                          description="搜索关键词 (用于 search)", required=False),
            ToolParameter(name="note_id", type="string",
                          description="笔记ID (用于 read/update/delete)", required=False),
            ToolParameter(name="limit", type="string",
                          description="返回数量限制 (默认 10)", required=False, default="10"),
        ]

    def run(self, parameters: Dict[str, Any]) -> str:
        """执行笔记操作"""
        action = parameters.get("action", "list")

        try:
            if action == "create":
                return self._create(parameters)
            elif action == "read":
                return self._read(parameters)
            elif action == "update":
                return self._update(parameters)
            elif action == "search":
                return self._search(parameters)
            elif action == "list":
                return self._list(parameters)
            elif action == "summary":
                return self._summary()
            elif action == "delete":
                return self._delete(parameters)
            else:
                return f"❌ 不支持的操作: {action}"
        except Exception as e:
            return f"❌ 笔记操作失败: {e}"

    # ========== 便捷方法 ==========

    def execute(self, action: str, **kwargs) -> str:
        kwargs["action"] = action
        return self.run(kwargs)

    # ========== 核心操作 ==========

    def _create(self, params: Dict[str, Any]) -> str:
        title = params.get("title", "")
        content = params.get("content", "")
        if not title:
            return "❌ 创建笔记失败: 标题不能为空"

        note_type = params.get("note_type", "general")
        tags = self._parse_tags(params.get("tags", ""))

        # 生成 ID
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        note_id = f"note_{ts}_{len(self.index)}"

        metadata = {
            "id": note_id, "title": title, "type": note_type,
            "tags": tags,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        # 构建 Markdown
        md = self._build_markdown(metadata, content)
        file_path = os.path.join(self.workspace, f"{note_id}.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md)

        metadata["file_path"] = file_path
        self.index[note_id] = metadata
        self._save_index()

        return f"✅ 笔记已创建: {title} (ID: {note_id}, 类型: {note_type})"

    def _read(self, params: Dict[str, Any]) -> str:
        note_id = params.get("note_id", "")
        if note_id not in self.index:
            return f"❌ 笔记不存在: {note_id}"

        note = self._load_note(note_id)
        metadata = note["metadata"]
        content = note["content"]

        return (
            f"📄 笔记: {metadata.get('title', '')}\n"
            f"   ID: {note_id}\n"
            f"   类型: {metadata.get('type', 'general')}\n"
            f"   标签: {', '.join(metadata.get('tags', []))}\n"
            f"   更新: {metadata.get('updated_at', '')}\n"
            f"---\n{content}"
        )

    def _update(self, params: Dict[str, Any]) -> str:
        note_id = params.get("note_id", "")
        if note_id not in self.index:
            return f"❌ 笔记不存在: {note_id}"

        note = self._load_note(note_id)
        metadata = note["metadata"]
        old_content = note["content"]

        if "title" in params and params["title"]:
            metadata["title"] = params["title"]
        if "note_type" in params:
            metadata["type"] = params["note_type"]
        if "tags" in params:
            metadata["tags"] = self._parse_tags(params["tags"])
        if "content" in params:
            old_content = params["content"]

        metadata["updated_at"] = datetime.now().isoformat()

        md = self._build_markdown(metadata, old_content)
        with open(metadata["file_path"], "w", encoding="utf-8") as f:
            f.write(md)

        self.index[note_id] = metadata
        self._save_index()

        return f"✅ 笔记已更新: {metadata['title']}"

    def _search(self, params: Dict[str, Any]) -> str:
        query = params.get("query", "")
        limit = int(params.get("limit", 10))
        note_type = params.get("note_type")
        tags = self._parse_tags(params.get("tags", ""))

        if not query:
            return self._list(params)

        results = []
        query_lower = query.lower()

        for note_id, meta in self.index.items():
            if note_type and meta.get("type") != note_type:
                continue
            if tags and not set(tags).intersection(set(meta.get("tags", []))):
                continue

            try:
                note = self._load_note(note_id)
                title = meta.get("title", "")
                content = note["content"]
                if query_lower in title.lower() or query_lower in content.lower():
                    results.append({
                        "note_id": note_id,
                        "title": title,
                        "type": meta.get("type"),
                        "tags": meta.get("tags", []),
                        "content_preview": content[:200],
                        "updated_at": meta.get("updated_at"),
                    })
            except Exception:
                continue

        results.sort(key=lambda x: x["updated_at"] or "", reverse=True)
        results = results[:limit]

        if not results:
            return f"🔍 未找到与 '{query}' 相关的笔记"

        lines = [f"🔍 找到 {len(results)} 条相关笔记:"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. [{r['type']}] {r['title']} "
                f"(标签: {', '.join(r['tags'])})"
            )
        return "\n".join(lines)

    def _list(self, params: Dict[str, Any]) -> str:
        note_type = params.get("note_type")
        tags = self._parse_tags(params.get("tags", ""))
        limit = int(params.get("limit", 20))

        results = []
        for note_id, meta in self.index.items():
            if note_type and meta.get("type") != note_type:
                continue
            if tags and not set(tags).intersection(set(meta.get("tags", []))):
                continue
            results.append(meta)

        results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        results = results[:limit]

        if not results:
            return "📝 暂无笔记"

        lines = [f"📝 笔记列表 ({len(results)} 条):"]
        for i, meta in enumerate(results, 1):
            lines.append(
                f"{i}. [{meta.get('type', 'general')}] {meta.get('title', '无标题')} "
                f"(ID: {meta['id']})"
            )
        return "\n".join(lines)

    def _summary(self) -> str:
        total = len(self.index)
        if total == 0:
            return "📝 暂无笔记"

        type_counts: Dict[str, int] = {}
        for meta in self.index.values():
            t = meta.get("type", "general")
            type_counts[t] = type_counts.get(t, 0) + 1

        recent = sorted(
            self.index.values(),
            key=lambda x: x.get("updated_at", ""), reverse=True,
        )[:5]

        lines = ["📊 笔记摘要:", f"  总计: {total} 条"]
        lines.append("  类型分布:")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    - {t}: {c}")
        lines.append("  最近更新:")
        for meta in recent:
            lines.append(
                f"    - [{meta.get('type', '?')}] {meta.get('title', '无标题')}"
            )
        return "\n".join(lines)

    def _delete(self, params: Dict[str, Any]) -> str:
        note_id = params.get("note_id", "")
        if note_id not in self.index:
            return f"❌ 笔记不存在: {note_id}"

        file_path = self.index[note_id].get("file_path", "")
        if os.path.exists(file_path):
            os.remove(file_path)

        title = self.index[note_id].get("title", note_id)
        del self.index[note_id]
        self._save_index()

        return f"🗑️ 笔记已删除: {title}"

    # ========== 内部方法 ==========

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self._index_path):
            try:
                with open(self._index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_index(self):
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

    def _load_note(self, note_id: str) -> Dict[str, Any]:
        file_path = self.index[note_id]["file_path"]
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()
        return self._parse_markdown(raw)

    def _parse_markdown(self, raw: str) -> Dict[str, Any]:
        """解析 Markdown, 分离 YAML 和正文"""
        parts = raw.split("---\n", 2)
        if len(parts) >= 3 and _has_yaml:
            try:
                metadata = yaml.safe_load(parts[1]) or {}
            except Exception:
                metadata = self._simple_yaml_parse(parts[1])
            content = parts[2].strip()
        elif len(parts) >= 3:
            metadata = self._simple_yaml_parse(parts[1])
            content = parts[2].strip()
        else:
            metadata = {}
            content = raw.strip()
        return {"metadata": metadata, "content": content}

    def _simple_yaml_parse(self, yaml_str: str) -> Dict[str, Any]:
        """简单 YAML 解析 (无需 pyyaml)"""
        result: Dict[str, Any] = {}
        for line in yaml_str.splitlines():
            line = line.strip()
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                # 列表
                if val.startswith("[") and val.endswith("]"):
                    val = [v.strip().strip('"').strip("'") for v in val[1:-1].split(",")]
                result[key] = val
        return result

    def _build_markdown(self, metadata: Dict[str, Any], content: str) -> str:
        """构建 Markdown (YAML 元数据 + 正文)"""
        if _has_yaml:
            yaml_str = yaml.dump(metadata, allow_unicode=True, sort_keys=False, default_flow_style=False)
        else:
            yaml_str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
        return f"---\n{yaml_str}---\n\n{content}"

    def _parse_tags(self, tags_input: Any) -> List[str]:
        if isinstance(tags_input, list):
            return tags_input
        if isinstance(tags_input, str) and tags_input.strip():
            return [t.strip() for t in tags_input.split(",") if t.strip()]
        return []

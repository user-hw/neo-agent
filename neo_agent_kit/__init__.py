"""neo-agent-kit: 一个轻量级、教学友好的 AI Agent 框架

特性:
- 4 种 Agent 范式: SimpleAgent / ReActAgent / ReflectionAgent / PlanAndSolveAgent
- 记忆系统: WorkingMemory / EpisodicMemory / SemanticMemory
- RAG 系统: 文档处理 + 智能检索 + LLM 增强生成
- 多提供商 LLM 支持: OpenAI / DeepSeek / 智谱 / ModelScope / Ollama / VLLM
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.2.0"

__all__ = [
    # LLM & Core
    "NeoAgentLLM",
    "Message",
    "MessageRole",
    "Config",
    "Agent",
    # Exceptions
    "NeoAgentError",
    "LLMError",
    "ToolError",
    "ConfigError",
    "AgentError",
    # Tools
    "Tool",
    "ToolParameter",
    "ToolRegistry",
    # Agents
    "SimpleAgent",
    "ReActAgent",
    "ReflectionAgent",
    "PlanAndSolveAgent",
    # Memory
    "MemoryManager",
    "MemoryConfig",
    "MemoryItem",
    # RAG
    "RAGPipeline",
    "DocumentProcessor",
]

_EXPORTS = {
    # Core
    "NeoAgentLLM": ("neo_agent_kit.core.llm", "NeoAgentLLM"),
    "Message": ("neo_agent_kit.core.message", "Message"),
    "MessageRole": ("neo_agent_kit.core.message", "MessageRole"),
    "Config": ("neo_agent_kit.core.config", "Config"),
    "Agent": ("neo_agent_kit.core.agent", "Agent"),
    # Exceptions
    "NeoAgentError": ("neo_agent_kit.core.exceptions", "NeoAgentError"),
    "LLMError": ("neo_agent_kit.core.exceptions", "LLMError"),
    "ToolError": ("neo_agent_kit.core.exceptions", "ToolError"),
    "ConfigError": ("neo_agent_kit.core.exceptions", "ConfigError"),
    "AgentError": ("neo_agent_kit.core.exceptions", "AgentError"),
    # Tools
    "Tool": ("neo_agent_kit.tools.base", "Tool"),
    "ToolParameter": ("neo_agent_kit.tools.base", "ToolParameter"),
    "ToolRegistry": ("neo_agent_kit.tools.registry", "ToolRegistry"),
    # Agents
    "SimpleAgent": ("neo_agent_kit.agents.simple_agent", "SimpleAgent"),
    "ReActAgent": ("neo_agent_kit.agents.react_agent", "ReActAgent"),
    "ReflectionAgent": ("neo_agent_kit.agents.reflection_agent", "ReflectionAgent"),
    "PlanAndSolveAgent": ("neo_agent_kit.agents.plan_solve_agent", "PlanAndSolveAgent"),
    # Memory
    "MemoryManager": ("neo_agent_kit.memory.manager", "MemoryManager"),
    "MemoryConfig": ("neo_agent_kit.memory.base", "MemoryConfig"),
    "MemoryItem": ("neo_agent_kit.memory.base", "MemoryItem"),
    # RAG
    "RAGPipeline": ("neo_agent_kit.rag.pipeline", "RAGPipeline"),
    "DocumentProcessor": ("neo_agent_kit.rag.document", "DocumentProcessor"),
}


def __getattr__(name: str):
    """Lazily load public exports to avoid importing heavy deps too early."""
    if name not in _EXPORTS:
        raise AttributeError(f"module 'neo_agent_kit' has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


if TYPE_CHECKING:
    from .agents.plan_solve_agent import PlanAndSolveAgent
    from .agents.react_agent import ReActAgent
    from .agents.reflection_agent import ReflectionAgent
    from .agents.simple_agent import SimpleAgent
    from .core.agent import Agent
    from .core.config import Config
    from .core.exceptions import AgentError, ConfigError, LLMError, NeoAgentError, ToolError
    from .core.llm import NeoAgentLLM
    from .core.message import Message, MessageRole
    from .memory.base import MemoryConfig, MemoryItem
    from .memory.manager import MemoryManager
    from .rag.document import DocumentProcessor
    from .rag.pipeline import RAGPipeline
    from .tools.base import Tool, ToolParameter
    from .tools.registry import ToolRegistry

"""neo-agent: 一个轻量级、教学友好的 AI Agent 框架"""

# 核心组件
from .core.llm import NeoAgentLLM
from .core.message import Message, MessageRole
from .core.config import Config
from .core.agent import Agent
from .core.exceptions import (
    NeoAgentError,
    LLMError,
    ToolError,
    ConfigError,
    AgentError,
)

# 工具系统
from .tools.base import Tool, ToolParameter
from .tools.registry import ToolRegistry

# Agent 实现
from .agents.simple_agent import SimpleAgent
from .agents.react_agent import ReActAgent
from .agents.reflection_agent import ReflectionAgent
from .agents.plan_solve_agent import PlanAndSolveAgent

__version__ = "0.1.0"
__all__ = [
    # 核心
    "NeoAgentLLM",
    "Message",
    "MessageRole",
    "Config",
    "Agent",
    # 异常
    "NeoAgentError",
    "LLMError",
    "ToolError",
    "ConfigError",
    "AgentError",
    # 工具
    "Tool",
    "ToolParameter",
    "ToolRegistry",
    # Agent
    "SimpleAgent",
    "ReActAgent",
    "ReflectionAgent",
    "PlanAndSolveAgent",
]

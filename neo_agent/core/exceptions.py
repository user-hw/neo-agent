"""neo-agent 异常体系"""


class NeoAgentError(Exception):
    """neo-agent 基础异常"""
    pass


class LLMError(NeoAgentError):
    """LLM 调用相关异常"""
    pass


class ToolError(NeoAgentError):
    """工具执行相关异常"""
    pass


class ConfigError(NeoAgentError):
    """配置相关异常"""
    pass


class AgentError(NeoAgentError):
    """Agent 执行相关异常"""
    pass

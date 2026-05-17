"""配置管理"""
import os
from typing import Optional, Dict, Any
from pydantic import BaseModel


class Config(BaseModel):
    """neo-agent-kit 配置类

    集中管理框架所有配置项，支持从环境变量读取。
    """

    # LLM 配置
    default_model: str = "gpt-3.5-turbo"
    default_provider: str = "auto"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    # 系统配置
    debug: bool = False
    log_level: str = "INFO"

    # Agent 配置
    max_history_length: int = 100
    max_tool_iterations: int = 5
    max_react_steps: int = 5
    max_reflection_rounds: int = 3

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置"""
        return cls(
            default_model=os.getenv("LLM_MODEL_ID", "gpt-3.5-turbo"),
            default_provider=os.getenv("LLM_PROVIDER", "auto"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS")) if os.getenv("LLM_MAX_TOKENS") else None,
            debug=os.getenv("NEO_DEBUG", "false").lower() == "true",
            log_level=os.getenv("NEO_LOG_LEVEL", "INFO"),
            max_history_length=int(os.getenv("NEO_MAX_HISTORY", "100")),
            max_tool_iterations=int(os.getenv("NEO_MAX_TOOL_ITERATIONS", "5")),
            max_react_steps=int(os.getenv("NEO_MAX_REACT_STEPS", "5")),
            max_reflection_rounds=int(os.getenv("NEO_MAX_REFLECTION_ROUNDS", "3")),
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.model_dump()

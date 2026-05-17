"""Agent 抽象基类"""
from abc import ABC, abstractmethod
from typing import Optional, List
from .message import Message
from .llm import NeoAgentLLM
from .config import Config


class Agent(ABC):
    """Agent 抽象基类

    所有具体的 Agent 实现都必须继承此类并实现 run 方法。
    """

    def __init__(
        self,
        name: str,
        llm: NeoAgentLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: List[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行 Agent

        Args:
            input_text: 用户输入文本

        Returns:
            Agent 响应文本
        """
        pass

    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self._history.append(message)

    def clear_history(self):
        """清空历史记录"""
        self._history.clear()

    def get_history(self) -> List[Message]:
        """获取历史记录副本"""
        return self._history.copy()

    def _build_messages(self, input_text: str, include_system: bool = True) -> list:
        """构建发送给 LLM 的消息列表

        Args:
            input_text: 用户输入
            include_system: 是否包含系统消息

        Returns:
            OpenAI 格式的消息列表
        """
        messages = []

        if include_system and self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for msg in self._history:
            messages.append(msg.to_dict())

        messages.append({"role": "user", "content": input_text})
        return messages

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"

    def __repr__(self) -> str:
        return self.__str__()

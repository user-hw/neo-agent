"""消息系统"""
from typing import Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel

# 定义消息角色的类型，限制其取值
MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """消息类 - 框架内统一的消息格式"""

    content: str
    role: MessageRole
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None

    def __init__(self, content: str, role: MessageRole, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get('timestamp', datetime.now()),
            metadata=kwargs.get('metadata', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（兼容 OpenAI API）"""
        result = {
            "role": self.role,
            "content": self.content
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"

    def __repr__(self) -> str:
        return self.__str__()

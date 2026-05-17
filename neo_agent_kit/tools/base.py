"""工具基类与参数定义"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel


class ToolParameter(BaseModel):
    """工具参数定义"""
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None


class Tool(ABC):
    """工具抽象基类

    所有工具都必须继承此类并实现 run 和 get_parameters 方法。
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str:
        """执行工具

        Args:
            parameters: 参数字典

        Returns:
            执行结果字符串
        """
        pass

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数定义"""
        pass

    def to_openai_schema(self) -> Dict[str, Any]:
        """转换为 OpenAI function calling schema

        Returns:
            符合 OpenAI function calling 标准的 schema
        """
        parameters = self.get_parameters()
        properties = {}
        required = []

        for param in parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.default is not None:
                prop["description"] = f"{param.description} (默认: {param.default})"
            if param.type == "array":
                prop["items"] = {"type": "string"}
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    def __repr__(self) -> str:
        return f"Tool(name={self.name})"

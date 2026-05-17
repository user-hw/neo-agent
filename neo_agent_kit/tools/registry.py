"""工具注册机制"""
from typing import Dict, Any, Callable, List, Optional
from .base import Tool


class ToolRegistry:
    """工具注册表 - 工具系统的管理中枢

    支持两种注册方式:
    1. Tool 对象注册 - 适合复杂工具
    2. 函数直接注册 - 适合简单工具
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._functions: Dict[str, Dict[str, Any]] = {}

    # ========== 注册方法 ==========

    def register_tool(self, tool: Tool):
        """注册 Tool 对象"""
        if tool.name in self._tools:
            print(f"⚠️ 警告: 工具 '{tool.name}' 已存在，将被覆盖。")
        self._tools[tool.name] = tool
        print(f"✅ 工具 '{tool.name}' 已注册。")

    def register_function(
        self,
        name: str,
        description: str,
        func: Callable[[str], str]
    ):
        """注册函数作为工具（简便方式）

        Args:
            name: 工具名称
            description: 工具描述
            func: 工具函数，接受字符串参数，返回字符串结果
        """
        if name in self._functions:
            print(f"⚠️ 警告: 工具 '{name}' 已存在，将被覆盖。")
        self._functions[name] = {
            "description": description,
            "func": func
        }
        print(f"✅ 工具 '{name}' 已注册。")

    # ========== 查询方法 ==========

    def get_tool(self, name: str) -> Optional[Tool]:
        """获取指定的 Tool 对象"""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """列出所有已注册的工具名称"""
        tools = list(self._tools.keys()) + list(self._functions.keys())
        return tools

    def get_tools_description(self) -> str:
        """获取所有可用工具的格式化描述"""
        descriptions = []

        for tool in self._tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")

        for name, info in self._functions.items():
            descriptions.append(f"- {name}: {info['description']}")

        return "\n".join(descriptions) if descriptions else "暂无可用工具"

    def to_openai_schemas(self) -> List[Dict[str, Any]]:
        """将所有 Tool 对象转换为 OpenAI schemas"""
        schemas = []
        for tool in self._tools.values():
            try:
                schemas.append(tool.to_openai_schema())
            except Exception as e:
                print(f"⚠️ 工具 '{tool.name}' schema 转换失败: {e}")
        return schemas

    # ========== 执行方法 ==========

    def execute_tool(self, name: str, input_data: str) -> str:
        """执行指定的工具

        Args:
            name: 工具名称
            input_data: 输入数据（字符串形式）

        Returns:
            工具执行结果
        """
        # 优先查找 Tool 对象
        if name in self._tools:
            tool = self._tools[name]
            try:
                # 尝试解析参数
                params = self._parse_input(tool, input_data)
                return tool.run(params)
            except Exception as e:
                return f"❌ 工具 '{name}' 执行失败: {e}"

        # 查找注册的函数
        if name in self._functions:
            try:
                return self._functions[name]["func"](input_data)
            except Exception as e:
                return f"❌ 函数工具 '{name}' 执行失败: {e}"

        return f"❌ 未找到工具 '{name}'"

    def _parse_input(self, tool: Tool, input_data: str) -> Dict[str, Any]:
        """智能解析工具输入为参数字典"""
        parameters = tool.get_parameters()

        # 如果只有一个参数，直接赋值
        if len(parameters) == 1:
            return {parameters[0].name: input_data}

        # 尝试 key=value 格式解析
        params = {}
        if '=' in input_data:
            pairs = input_data.split(',')
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    params[key.strip()] = value.strip()
            if params:
                return params

        # 回退：第一个参数接收全部输入
        if parameters:
            params[parameters[0].name] = input_data
        return params

    # ========== 管理方法 ==========

    def unregister(self, name: str) -> bool:
        """移除工具"""
        removed = False
        if name in self._tools:
            del self._tools[name]
            removed = True
        if name in self._functions:
            del self._functions[name]
            removed = True
        if removed:
            print(f"🗑️ 工具 '{name}' 已移除。")
        return removed

    def clear(self):
        """清空所有工具"""
        self._tools.clear()
        self._functions.clear()
        print("🗑️ 所有工具已清空。")

    def __len__(self) -> int:
        return len(self._tools) + len(self._functions)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={len(self)})"

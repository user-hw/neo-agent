"""SimpleAgent - 基础对话智能体

支持基础对话和可选工具调用。
"""
import re
from typing import Optional, Iterator
from ..core.agent import Agent
from ..core.llm import NeoAgentLLM
from ..core.message import Message
from ..core.config import Config
from ..tools.registry import ToolRegistry


class SimpleAgent(Agent):
    """基础对话智能体

    特性:
    - 基础对话能力
    - 可选工具调用（通过 TOOL_CALL 标记）
    - 流式响应支持
    - 对话历史管理
    """

    def __init__(
        self,
        name: str,
        llm: NeoAgentLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional[ToolRegistry] = None,
        enable_tool_calling: bool = True
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        print(f"✅ {name} 初始化完成，工具调用: {'启用' if self.enable_tool_calling else '禁用'}")

    # ========== 核心运行方法 ==========

    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """运行 SimpleAgent

        Args:
            input_text: 用户输入
            max_tool_iterations: 最大工具调用轮数

        Returns:
            Agent 响应
        """
        print(f"🤖 {self.name} 正在处理: {input_text}")

        # 构建消息列表
        messages = self._build_enhanced_messages(input_text)

        # 无工具时使用简单对话
        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            print(f"✅ {self.name} 响应完成")
            return response

        # 支持多轮工具调用
        return self._run_with_tools(messages, input_text, max_tool_iterations, **kwargs)

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """流式运行 - 实时返回响应

        Args:
            input_text: 用户输入

        Yields:
            响应文本的每个 chunk
        """
        print(f"🌊 {self.name} 开始流式处理: {input_text}")

        messages = self._build_enhanced_messages(input_text)
        full_response = ""
        print("📝 实时响应: ", end="")

        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            print(chunk, end="", flush=True)
            yield chunk

        print()
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))
        print(f"✅ {self.name} 流式响应完成")

    # ========== 工具调用逻辑 ==========

    def _run_with_tools(
        self,
        messages: list,
        input_text: str,
        max_tool_iterations: int,
        **kwargs
    ) -> str:
        """支持工具调用的运行逻辑"""
        iteration = 0
        final_response = ""

        while iteration < max_tool_iterations:
            response = self.llm.invoke(messages, **kwargs)
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                print(f"🔧 检测到 {len(tool_calls)} 个工具调用")
                clean_response = response
                tool_results = []

                for call in tool_calls:
                    result = self._execute_tool_call(call['tool_name'], call['parameters'])
                    tool_results.append(result)
                    clean_response = clean_response.replace(call['original'], "")

                messages.append({"role": "assistant", "content": clean_response})
                messages.append({
                    "role": "user",
                    "content": f"工具执行结果:\n{chr(10).join(tool_results)}\n\n请基于这些结果给出完整的回答。"
                })
                iteration += 1
                continue

            final_response = response
            break

        if iteration >= max_tool_iterations and not final_response:
            final_response = self.llm.invoke(messages, **kwargs)

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))
        print(f"✅ {self.name} 响应完成")
        return final_response

    def _parse_tool_calls(self, text: str) -> list:
        """解析文本中的工具调用标记"""
        pattern = r'\[TOOL_CALL:([^:]+):([^\]]+)\]'
        matches = re.findall(pattern, text)
        return [
            {
                'tool_name': name.strip(),
                'parameters': params.strip(),
                'original': f'[TOOL_CALL:{name}:{params}]'
            }
            for name, params in matches
        ]

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        """执行单个工具调用"""
        if not self.tool_registry:
            return "❌ 错误: 未配置工具注册表"

        try:
            if tool_name == 'calculator':
                result = self.tool_registry.execute_tool(tool_name, parameters)
            else:
                param_dict = self._parse_tool_parameters(tool_name, parameters)
                tool = self.tool_registry.get_tool(tool_name)
                if not tool:
                    return f"❌ 错误: 未找到工具 '{tool_name}'"
                result = tool.run(param_dict)
            return f"🔧 工具 {tool_name} 执行结果:\n{result}"
        except Exception as e:
            return f"❌ 工具调用失败: {e}"

    def _parse_tool_parameters(self, tool_name: str, parameters: str) -> dict:
        """智能解析工具参数"""
        if '=' in parameters:
            pairs = parameters.split(',')
            return dict(
                (k.strip(), v.strip())
                for pair in pairs
                if '=' in pair
                for k, v in [pair.split('=', 1)]
            )
        # 根据工具类型智能推断
        mapping = {
            'search': {'query': parameters},
            'memory': {'action': 'search', 'query': parameters},
        }
        return mapping.get(tool_name, {'input': parameters})

    # ========== 辅助方法 ==========

    def _build_enhanced_messages(self, input_text: str) -> list:
        """构建增强的消息列表（包含工具信息）"""
        messages = []

        if self.system_prompt:
            enhanced = self._get_enhanced_system_prompt()
            messages.append({"role": "system", "content": enhanced})

        for msg in self._history:
            messages.append(msg.to_dict())

        messages.append({"role": "user", "content": input_text})
        return messages

    def _get_enhanced_system_prompt(self) -> str:
        """构建包含工具信息的系统提示词"""
        base = self.system_prompt or "你是一个有用的AI助手。"

        if not self.enable_tool_calling or not self.tool_registry:
            return base

        tools_desc = self.tool_registry.get_tools_description()
        if not tools_desc or tools_desc == "暂无可用工具":
            return base

        return (
            f"{base}\n\n"
            f"## 可用工具\n"
            f"你可以使用以下工具来帮助回答问题:\n"
            f"{tools_desc}\n\n"
            f"## 工具调用格式\n"
            f"当需要使用工具时，请使用以下格式:\n"
            f"`[TOOL_CALL:工具名:参数]`\n"
            f"例如: `[TOOL_CALL:search:Python编程]`\n\n"
            f"工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。\n"
        )

    # ========== 工具管理便利方法 ==========

    def add_tool(self, tool) -> None:
        """动态添加工具"""
        if not self.tool_registry:
            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True
        self.tool_registry.register_tool(tool)
        print(f"🔧 工具 '{tool.name}' 已添加到 Agent")

    def remove_tool(self, tool_name: str) -> bool:
        """移除工具"""
        if self.tool_registry:
            return self.tool_registry.unregister(tool_name)
        return False

    def has_tools(self) -> bool:
        """是否有可用工具"""
        return self.enable_tool_calling and self.tool_registry is not None

    def list_tools(self) -> list:
        """列出所有可用工具"""
        return self.tool_registry.list_tools() if self.tool_registry else []

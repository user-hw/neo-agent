"""ReActAgent - 推理与行动结合的智能体

ReAct = Reasoning (推理) + Acting (行动)
通过 "思考 → 行动 → 观察" 循环来解决问题。
"""
import re
from typing import Optional, List
from ..core.agent import Agent
from ..core.llm import NeoAgentLLM
from ..core.message import Message
from ..core.config import Config
from ..tools.registry import ToolRegistry

# 默认 ReAct 提示词模板
DEFAULT_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤:

Thought: 分析当前问题，思考需要什么信息或采取什么行动。
Action: 选择一个行动，格式必须是以下之一:
- `{tool_name}[{tool_input}]` - 调用指定工具
- `Finish[最终答案]` - 当你有足够信息给出最终答案时

## 重要提醒
1. 每次回应必须包含 Thought 和 Action 两部分
2. 工具调用的格式必须严格遵循: 工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用 Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动:
"""


class ReActAgent(Agent):
    """ReAct 智能体 - 推理与行动结合

    特性:
    - "思考→行动→观察" 循环
    - 严格的格式约束确保输出可解析
    - 可配置的最大步数
    - 支持自定义提示词模板
    """

    def __init__(
        self,
        name: str,
        llm: NeoAgentLLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: List[str] = []
        self.prompt_template = custom_prompt or DEFAULT_REACT_PROMPT
        print(f"✅ {name} 初始化完成，最大步数: {max_steps}")

    def run(self, input_text: str, **kwargs) -> str:
        """运行 ReAct Agent

        Args:
            input_text: 用户问题

        Returns:
            最终答案
        """
        self.current_history = []
        current_step = 0

        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            # 1. 构建提示词
            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(self.current_history) if self.current_history else "（尚无执行历史）"
            prompt = self.prompt_template.format(
                tools=tools_desc,
                question=input_text,
                history=history_str
            )

            # 2. 调用 LLM
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm.invoke(messages, **kwargs)
            print(f"📥 模型输出:\n{response_text[:300]}...")

            # 3. 解析输出
            thought, action = self._parse_output(response_text)

            if thought:
                print(f"💭 思考: {thought[:100]}...")

            # 4. 检查完成条件
            if action and action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(final_answer, "assistant"))
                print(f"✅ {self.name} 完成任务")
                return final_answer

            # 5. 执行工具调用
            if action:
                tool_name, tool_input = self._parse_action(action)
                if tool_name and tool_input:
                    print(f"🔧 执行: {tool_name}[{tool_input}]")
                    observation = self.tool_registry.execute_tool(tool_name, tool_input)
                    print(f"👁️ 观察结果: {observation[:100]}...")
                    self.current_history.append(f"Step {current_step}:")
                    self.current_history.append(f"Action: {action}")
                    self.current_history.append(f"Observation: {observation}")
                else:
                    self.current_history.append(f"Step {current_step}: 无法解析 Action")
            else:
                self.current_history.append(f"Step {current_step}: 未找到有效 Action")

        # 达到最大步数
        final_answer = f"抱歉，我无法在 {self.max_steps} 步内完成这个任务。"
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer

    def _parse_output(self, text: str) -> tuple:
        """解析 LLM 输出，提取 Thought 和 Action"""
        thought = None
        action = None

        # 提取 Thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=\nAction:|\Z)', text, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()

        # 提取 Action
        action_match = re.search(r'Action:\s*(.+)', text, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()

        return thought, action

    def _parse_action(self, action_text: str) -> tuple:
        """解析 Action 文本，提取工具名和输入

        支持格式:
        - tool_name[input]
        - Finish[answer]
        """
        # 匹配 tool_name[input] 或 Finish[answer]
        match = re.match(r'(\w+)\[(.+)\]', action_text.strip())
        if match:
            return match.group(1), match.group(2).rstrip(']')
        return None, None

    def _parse_action_input(self, action_text: str) -> str:
        """从 Finish[answer] 中提取最终答案"""
        match = re.match(r'Finish\[(.+)\]', action_text.strip())
        if match:
            return match.group(1).rstrip(']')
        return action_text

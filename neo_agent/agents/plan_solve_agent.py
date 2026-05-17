"""PlanAndSolveAgent - 规划与执行分离的智能体

先制定计划，再逐步执行。
"""
import json
import re
from typing import Optional, List, Dict
from ..core.agent import Agent
from ..core.llm import NeoAgentLLM
from ..core.message import Message
from ..core.config import Config

# 默认提示词模板
DEFAULT_PLANNER_PROMPT = """你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

DEFAULT_EXECUTOR_PROMPT = """你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""


class PlanAndSolveAgent(Agent):
    """规划与执行分离智能体

    工作流程:
    1. Planner: 将复杂问题分解为子步骤列表
    2. Executor: 逐步执行每个子步骤

    特性:
    - 规划与执行分离，职责清晰
    - 强制结构化的计划输出（Python 列表）
    - 历史步骤信息传递给后续执行
    - 支持自定义提示词模板
    """

    def __init__(
        self,
        name: str,
        llm: NeoAgentLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[Dict[str, str]] = None
    ):
        super().__init__(name, llm, system_prompt, config)
        self.planner_prompt = (custom_prompts or {}).get("planner", DEFAULT_PLANNER_PROMPT)
        self.executor_prompt = (custom_prompts or {}).get("executor", DEFAULT_EXECUTOR_PROMPT)
        print(f"✅ {name} 初始化完成")

    def run(self, input_text: str, **kwargs) -> str:
        """运行 PlanAndSolveAgent

        Args:
            input_text: 用户问题

        Returns:
            最终答案
        """
        print(f"\n🤖 {self.name} 开始处理问题: {input_text[:80]}...")

        # 阶段1: 规划
        print("📋 阶段 1: 制定计划...")
        plan = self._plan(input_text, **kwargs)
        steps = self._parse_plan(plan)

        if not steps:
            print("⚠️ 计划解析失败，使用单步执行")
            steps = [input_text]

        print(f"📋 计划共 {len(steps)} 步:")
        for i, step in enumerate(steps, 1):
            print(f"   {i}. {step}")

        # 阶段2: 执行
        print("⚙️ 阶段 2: 逐步执行...")
        history = []
        final_result = ""

        for i, step in enumerate(steps, 1):
            print(f"\n--- 执行步骤 {i}/{len(steps)}: {step[:50]}... ---")
            result = self._execute_step(
                question=input_text,
                plan=plan,
                history=history,
                current_step=step,
                **kwargs
            )
            history.append({"step": step, "result": result})
            final_result = result
            print(f"✅ 步骤 {i} 完成")

        # 组装最终答案
        if len(steps) > 1:
            summary_prompt = (
                f"请根据以下各步骤的结果，给出完整答案:\n\n"
                f"问题: {input_text}\n\n"
                + "\n".join(f"步骤 {i+1}: {h['step']}\n结果: {h['result']}\n"
                           for i, h in enumerate(history))
            )
            messages = [{"role": "user", "content": summary_prompt}]
            final_result = self.llm.invoke(messages, **kwargs)

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_result, "assistant"))
        print(f"✅ {self.name} 完成任务")
        return final_result

    def _plan(self, question: str, **kwargs) -> str:
        """生成执行计划"""
        prompt = self.planner_prompt.format(question=question)
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, temperature=0.3, **kwargs)

    def _parse_plan(self, plan_text: str) -> List[str]:
        """解析计划文本，提取步骤列表

        支持格式:
        - Python 列表: ["步骤1", "步骤2"]
        - 编号列表: 1. 步骤1\n2. 步骤2
        - Markdown 列表: - 步骤1\n- 步骤2
        """
        # 尝试 Python 列表格式
        code_match = re.search(r'```(?:python)?\s*(\[.*?\])\s*```', plan_text, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试直接解析 [...] 格式
        list_match = re.search(r'\[(.*?)\]', plan_text, re.DOTALL)
        if list_match:
            try:
                return json.loads(f"[{list_match.group(1)}]")
            except json.JSONDecodeError:
                pass

        # 回退: 按行解析
        lines = plan_text.strip().split('\n')
        steps = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 移除编号或项目符号
            cleaned = re.sub(r'^[\d]+[\.\)、]\s*', '', line)
            cleaned = re.sub(r'^[-*+]\s*', '', cleaned)
            if cleaned and len(cleaned) > 3:
                steps.append(cleaned.strip('"\''))

        return steps if steps else [plan_text]

    def _execute_step(
        self,
        question: str,
        plan: str,
        history: List[Dict],
        current_step: str,
        **kwargs
    ) -> str:
        """执行单个步骤"""
        history_str = "\n".join(
            f"步骤 {i+1}: {h['step']}\n结果: {h['result']}\n"
            for i, h in enumerate(history)
        ) if history else "（这是第一个步骤）"

        prompt = self.executor_prompt.format(
            question=question,
            plan=plan,
            history=history_str,
            current_step=current_step
        )
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, **kwargs)

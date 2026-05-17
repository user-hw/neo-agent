"""ReflectionAgent - 自我反思与迭代优化智能体

通过 "执行 → 反思 → 优化" 循环来提升输出质量。
"""
from typing import Optional, Dict
from ..core.agent import Agent
from ..core.llm import NeoAgentLLM
from ..core.message import Message
from ..core.config import Config

# 默认提示词模板
DEFAULT_PROMPTS = {
    "initial": """请根据以下要求完成任务:

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """请仔细审查以下回答，并找出可能的问题或改进空间:

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
""",
    "refine": """请根据反馈意见改进你的回答:

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
"""
}


class ReflectionAgent(Agent):
    """反思优化智能体

    通过 "执行→反思→优化" 循环迭代提升输出质量。

    特性:
    - 三轮迭代：初始生成 → 反思审阅 → 优化改进
    - 灵活的提前终止条件
    - 支持自定义提示词模板
    - 适用于写作、代码生成、分析等各种场景
    """

    def __init__(
        self,
        name: str,
        llm: NeoAgentLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_rounds: int = 3,
        custom_prompts: Optional[Dict[str, str]] = None
    ):
        super().__init__(name, llm, system_prompt, config)
        self.max_rounds = max_rounds
        self.prompts = custom_prompts or DEFAULT_PROMPTS
        print(f"✅ {name} 初始化完成，最大反思轮数: {max_rounds}")

    def run(self, input_text: str, **kwargs) -> str:
        """运行 ReflectionAgent

        Args:
            input_text: 任务描述

        Returns:
            优化后的最终回答
        """
        print(f"\n🤖 {self.name} 开始处理任务: {input_text[:80]}...")

        # 第1步: 初始生成
        print("📝 步骤 1/3: 生成初始回答...")
        content = self._generate(input_text, **kwargs)

        # 第2-N步: 反思+优化循环
        for round_num in range(1, self.max_rounds):
            print(f"🔍 步骤 {round_num + 1}/{self.max_rounds + 1}: 反思中...")
            feedback = self._reflect(input_text, content, **kwargs)

            if self._is_done(feedback):
                print("✅ 回答已满足要求，无需继续优化")
                break

            print(f"🔧 步骤 {round_num + 1}/{self.max_rounds + 1}: 优化中...")
            content = self._refine(input_text, content, feedback, **kwargs)

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(content, "assistant"))
        print(f"✅ {self.name} 完成优化")
        return content

    def _generate(self, task: str, **kwargs) -> str:
        """生成初始回答"""
        prompt = self.prompts["initial"].format(task=task)
        messages = [
            {"role": "system", "content": self.system_prompt or "你是一个专业的AI助手。"},
            {"role": "user", "content": prompt}
        ]
        return self.llm.invoke(messages, **kwargs)

    def _reflect(self, task: str, content: str, **kwargs) -> str:
        """反思当前回答的不足"""
        prompt = self.prompts["reflect"].format(task=task, content=content)
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, temperature=0.3, **kwargs)

    def _refine(self, task: str, last_attempt: str, feedback: str, **kwargs) -> str:
        """根据反馈优化回答"""
        prompt = self.prompts["refine"].format(
            task=task,
            last_attempt=last_attempt,
            feedback=feedback
        )
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, **kwargs)

    def _is_done(self, feedback: str) -> bool:
        """判断是否需要继续优化"""
        stop_phrases = ["无需改进", "无需优化", "已经很好", "没有问题", "no improvement needed"]
        feedback_lower = feedback.lower()
        return any(phrase in feedback_lower for phrase in stop_phrases)

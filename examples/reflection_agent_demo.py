"""ReflectionAgent 示例 - 自我反思与迭代优化"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from neo_agent import NeoAgentLLM, ReflectionAgent


def main():
    llm = NeoAgentLLM()

    # ---- 通用反思模式 ----
    print("=" * 50)
    print("  通用反思模式")
    print("=" * 50)

    agent = ReflectionAgent(
        name="写作助手",
        llm=llm,
        max_rounds=2
    )

    result = agent.run("写一段介绍机器学习的话（50字以内）")
    print(f"\n最终结果: {result}\n")

    # ---- 自定义提示词（代码审查模式）----
    print("=" * 50)
    print("  代码审查模式")
    print("=" * 50)

    code_prompts = {
        "initial": "你是Python专家，请编写一个函数来解决以下任务:\n{task}",
        "reflect": "审查以下代码的质量和效率:\n任务:{task}\n代码:{content}",
        "refine": "根据反馈优化代码:\n任务:{task}\n反馈:{feedback}"
    }

    code_agent = ReflectionAgent(
        name="代码审查助手",
        llm=llm,
        max_rounds=2,
        custom_prompts=code_prompts
    )

    result = code_agent.run("写一个函数计算斐波那契数列的第n项")
    print(f"\n最终结果: {result}\n")


if __name__ == "__main__":
    main()

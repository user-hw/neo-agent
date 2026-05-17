"""ReActAgent 示例 - 推理与行动结合"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from neo_agent import NeoAgentLLM, ReActAgent, ToolRegistry
from neo_agent.tools.builtin import CalculatorTool


def main():
    llm = NeoAgentLLM()

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    agent = ReActAgent(
        name="ReAct计算助手",
        llm=llm,
        tool_registry=registry,
        max_steps=5,
    )

    questions = [
        "计算 15 * 8 + 32",
        "计算 sqrt(144) + 2**4 的结果",
    ]

    for q in questions:
        print(f"\n{'='*50}")
        print(f"  问题: {q}")
        print("=" * 50)
        result = agent.run(q)
        print(f"\n📌 最终答案: {result}")


if __name__ == "__main__":
    main()

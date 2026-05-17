"""SimpleAgent 示例 - 基础对话与工具调用"""

from dotenv import load_dotenv
load_dotenv()

from neo_agent_kit import NeoAgentLLM, SimpleAgent
from neo_agent_kit.tools.builtin import CalculatorTool


def main():
    llm = NeoAgentLLM()

    # ---- 基础对话 ----
    print("=" * 50)
    print("  基础对话模式")
    print("=" * 50)

    agent = SimpleAgent(
        name="对话助手",
        llm=llm,
        system_prompt="你是一个友好的AI助手。"
    )

    response = agent.run("用一句话介绍人工智能")
    print(f"响应: {response}\n")

    # ---- 工具增强 ----
    print("=" * 50)
    print("  工具增强模式")
    print("=" * 50)

    agent.add_tool(CalculatorTool())
    response = agent.run("请用工具计算 2 ** 10 + 3 * 5")
    print(f"响应: {response}\n")

    # ---- 流式输出 ----
    print("=" * 50)
    print("  流式输出模式")
    print("=" * 50)

    for chunk in agent.stream_run("说一句鼓励的话"):
        pass
    print()


if __name__ == "__main__":
    main()

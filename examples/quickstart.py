"""
neo-agent-kit 快速体验示例

最简单的使用方式 - 只需配置 .env
"""
from dotenv import load_dotenv
load_dotenv()

from neo_agent_kit import NeoAgentLLM, SimpleAgent


def main():
    print("🚀 neo-agent-kit 快速体验\n")

    # 自动检测 provider，无需手动指定
    llm = NeoAgentLLM()

    # 创建 SimpleAgent
    agent = SimpleAgent(
        name="Neo助手",
        llm=llm,
        system_prompt="你是一个名为 Neo 的 AI 助手，用友好简洁的方式回答问题。"
    )

    # 对话
    questions = [
        "你好！请用一句话介绍你自己",
        "什么是人工智能？",
    ]

    for q in questions:
        print(f"\n👤 用户: {q}")
        response = agent.run(q)
        print(f"🤖 Neo: {response}")

    print(f"\n📊 对话历史: {len(agent.get_history())} 条消息")


if __name__ == "__main__":
    main()

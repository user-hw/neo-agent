"""
neo-agent-kit 快速体验示例

涵盖核心功能：LLM对话、记忆系统、RAG知识检索。
只需配置 .env 即可运行。
"""
from dotenv import load_dotenv
load_dotenv()

from neo_agent_kit import NeoAgentLLM, SimpleAgent, ToolRegistry
from neo_agent_kit.tools.builtin import CalculatorTool, MemoryTool, RAGTool


def demo_basic_chat(llm):
    """基础对话"""
    print("=" * 50)
    print("  1. 基础对话")
    print("=" * 50)

    agent = SimpleAgent(
        name="Neo助手",
        llm=llm,
        system_prompt="你是一个名为 Neo 的 AI 助手，用友好简洁的方式回答问题。"
    )

    for q in ["你好！请用一句话介绍你自己", "什么是人工智能？"]:
        print(f"\n👤 用户: {q}")
        response = agent.run(q)
        print(f"🤖 Neo: {response}")

    print(f"\n📊 对话历史: {len(agent.get_history())} 条消息")


def demo_memory():
    """记忆系统体验"""
    print("\n" + "=" * 50)
    print("  2. 记忆系统 🧠")
    print("=" * 50)

    memory_tool = MemoryTool(user_id="quickstart_user")

    # 添加不同类型的记忆
    print("\n📝 添加记忆...")
    print(memory_tool.execute("add",
        content="用户张三是一名Python开发者，专注于机器学习",
        memory_type="semantic", importance=0.8))
    print(memory_tool.execute("add",
        content="今天讨论了深度学习框架选型",
        memory_type="episodic", importance=0.6, event_type="discussion"))

    # 搜索记忆
    print("\n🔍 搜索 'Python开发者'...")
    print(memory_tool.execute("search", query="Python开发者"))

    # 记忆摘要
    print("\n📊 记忆摘要:")
    print(memory_tool.execute("summary"))

    # 遗忘不重要记忆
    print("\n🧹 遗忘不重要记忆...")
    print(memory_tool.execute("forget", strategy="importance_based", threshold="0.3"))


def demo_rag():
    """RAG 知识检索体验"""
    print("\n" + "=" * 50)
    print("  3. RAG 知识检索 📚")
    print("=" * 50)

    rag_tool = RAGTool(rag_namespace="quickstart")

    # 添加知识
    print("\n📝 添加知识到知识库...")
    print(rag_tool.execute("add_text",
        text="Python 由 Guido van Rossum 于1991年首次发布，设计哲学强调代码可读性。",
        document_id="python_info"))
    print(rag_tool.execute("add_text",
        text="RAG（检索增强生成）结合信息检索和文本生成，通过检索外部知识来增强LLM回答准确性。",
        document_id="rag_info"))

    # 搜索
    print("\n🔍 搜索 'Python 历史'...")
    print(rag_tool.execute("search", query="Python 的历史"))

    # 智能问答
    print("\n💡 智能问答: 什么是RAG？")
    print(rag_tool.execute("ask", question="什么是RAG技术？"))

    # 统计
    print("\n📊 知识库统计:")
    print(rag_tool.execute("stats"))


def demo_agent_with_tools(llm):
    """Agent + 工具集成"""
    print("\n" + "=" * 50)
    print("  4. Agent + 工具集成 🤖")
    print("=" * 50)

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    registry.register_tool(MemoryTool(user_id="agent_demo"))

    agent = SimpleAgent(
        name="全能助手",
        llm=llm,
        system_prompt="你是一个有记忆和计算能力的AI助手。需要时用 [TOOL_CALL:工具名:参数] 格式调用工具。",
        tool_registry=registry,
        enable_tool_calling=True,
    )

    response = agent.run("请记住我叫李四，然后告诉我 128 * 3 等于多少")
    print(f"🤖 Agent: {response}")


def main():
    print("🚀 neo-agent-kit 快速体验\n")

    llm = NeoAgentLLM()

    demo_basic_chat(llm)
    demo_memory()
    demo_rag()
    # demo_agent_with_tools(llm)  # 取消注释以体验 Agent+工具集成


if __name__ == "__main__":
    main()

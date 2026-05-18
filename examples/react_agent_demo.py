"""ReActAgent 示例 - 推理与行动结合，集成记忆与RAG"""

from dotenv import load_dotenv
load_dotenv()

from neo_agent_kit import NeoAgentLLM, ReActAgent, ToolRegistry
from neo_agent_kit.tools.builtin import CalculatorTool, MemoryTool, RAGTool


def main():
    llm = NeoAgentLLM()

    # ---- 基础 ReAct（计算器） ----
    print("=" * 50)
    print("  1. ReAct + 计算器")
    print("=" * 50)

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    agent = ReActAgent(
        name="ReAct计算助手",
        llm=llm,
        tool_registry=registry,
        max_steps=5,
    )

    for q in ["计算 15 * 8 + 32", "计算 sqrt(144) + 2**4 的结果"]:
        print(f"\n  问题: {q}")
        print("  " + "-" * 40)
        result = agent.run(q)
        print(f"  📌 最终答案: {result}")

    # ---- ReAct + 记忆 ----
    print("\n" + "=" * 50)
    print("  2. ReAct + 记忆 🧠")
    print("=" * 50)

    memory_tool = MemoryTool(user_id="react_demo")
    registry.register_tool(memory_tool)

    # 预先添加一些记忆
    memory_tool.execute("add",
        content="用户经常询问算法和数据结构问题", memory_type="semantic", importance=0.8)
    memory_tool.execute("add",
        content="上次讨论了快速排序和归并排序的对比", memory_type="episodic",
        importance=0.7, event_type="discussion")

    memory_agent = ReActAgent(
        name="ReAct记忆助手",
        llm=llm,
        tool_registry=registry,
        max_steps=5,
    )

    result = memory_agent.run("搜索一下我之前学过哪些算法相关的知识")
    print(f"📌 最终答案: {result}")

    # ---- ReAct + RAG ----
    print("\n" + "=" * 50)
    print("  3. ReAct + RAG 📚")
    print("=" * 50)

    rag_tool = RAGTool(rag_namespace="react_demo")
    registry.register_tool(rag_tool)

    # 预先添加知识
    rag_tool.execute("add_text",
        text="ReAct (Reasoning + Acting) 是一种AI Agent范式，它将推理(Thought)和行动(Action)交替进行。"
             "每次先思考需要什么信息，然后调用工具获取，最后给出答案。",
        document_id="react_concept")
    rag_tool.execute("add_text",
        text="neo-agent-kit 的 ReActAgent 支持配置 max_steps 来控制最大步数，默认5步。"
             "每一步包含 Thought 分析和 Action 执行。",
        document_id="neo_react")

    rag_agent = ReActAgent(
        name="ReAct知识助手",
        llm=llm,
        tool_registry=registry,
        max_steps=5,
    )

    result = rag_agent.run("什么是ReAct？在neo-agent-kit中如何使用？")
    print(f"📌 最终答案: {result}")


if __name__ == "__main__":
    main()

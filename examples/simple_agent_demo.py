"""SimpleAgent 示例 - 基础对话、工具调用、记忆与RAG集成"""

from dotenv import load_dotenv
load_dotenv()

from neo_agent_kit import NeoAgentLLM, SimpleAgent, ToolRegistry
from neo_agent_kit.tools.builtin import CalculatorTool, MemoryTool, RAGTool


def main():
    llm = NeoAgentLLM()

    # ---- 基础对话 ----
    print("=" * 50)
    print("  1. 基础对话模式")
    print("=" * 50)

    agent = SimpleAgent(
        name="对话助手",
        llm=llm,
        system_prompt="你是一个友好的AI助手。"
    )

    response = agent.run("用一句话介绍人工智能")
    print(f"响应: {response}\n")

    # ---- 工具增强（计算器） ----
    print("=" * 50)
    print("  2. 计算器工具")
    print("=" * 50)

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    tool_agent = SimpleAgent(
        name="计算助手",
        llm=llm,
        system_prompt="你可以使用计算器工具。需要计算时用 [TOOL_CALL:calculator:expression=算式] 格式。",
        tool_registry=registry,
        enable_tool_calling=True,
    )

    response = tool_agent.run("请用工具计算 2 ** 10 + 3 * 5")
    print(f"响应: {response}\n")

    # ---- 记忆工具集成 ----
    print("=" * 50)
    print("  3. 记忆工具 🧠")
    print("=" * 50)

    memory_tool = MemoryTool(user_id="simple_demo")

    # 直接使用记忆工具
    print(memory_tool.execute("add",
        content="用户偏好简洁的回答风格", memory_type="semantic", importance=0.8))
    print(memory_tool.execute("add",
        content="今天学习了SimpleAgent的使用方法", memory_type="episodic",
        importance=0.7, event_type="learning"))

    print("\n搜索记忆:")
    print(memory_tool.execute("search", query="用户偏好"))

    print("\n记忆统计:")
    print(memory_tool.execute("stats"))

    # 将记忆工具注册到Agent
    registry.register_tool(memory_tool)

    memory_agent = SimpleAgent(
        name="记忆助手",
        llm=llm,
        system_prompt="你是有记忆的AI助手。用 [TOOL_CALL:memory:action=add,content=xxx,memory_type=semantic] 格式记录信息。",
        tool_registry=registry,
        enable_tool_calling=True,
    )

    response = memory_agent.run("请记住：我的名字是王五，喜欢喝咖啡")
    print(f"\nAgent记忆响应: {response}")

    # ---- RAG 工具集成 ----
    print("\n" + "=" * 50)
    print("  4. RAG 知识检索 📚")
    print("=" * 50)

    rag_tool = RAGTool(rag_namespace="simple_demo")
    registry.register_tool(rag_tool)

    # 先添加知识
    rag_tool.execute("add_text",
        text="SimpleAgent是neo-agent-kit中最基础的智能体，支持对话和可选工具调用。"
             "它通过 TOOL_CALL 标记来识别和调用工具。",
        document_id="simple_agent_doc")

    rag_agent = SimpleAgent(
        name="知识助手",
        llm=llm,
        system_prompt="你是知识助手，使用RAG工具检索知识库。",
        tool_registry=registry,
        enable_tool_calling=True,
    )

    response = rag_agent.run("SimpleAgent 是什么？如何使用它？")
    print(f"Agent RAG响应: {response}")

    # ---- 流式输出 ----
    print("\n" + "=" * 50)
    print("  5. 流式输出")
    print("=" * 50)

    for chunk in agent.stream_run("用一句话鼓励编程初学者"):
        pass
    print()


if __name__ == "__main__":
    main()

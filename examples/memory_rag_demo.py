"""
neo-agent-kit 记忆与RAG体验示例

演示记忆系统和RAG工具的完整使用。
"""
from dotenv import load_dotenv
load_dotenv()

from neo_agent_kit import NeoAgentLLM, SimpleAgent, ToolRegistry
from neo_agent_kit.tools.builtin import MemoryTool, RAGTool, CalculatorTool


def demo_memory():
    """演示记忆系统"""
    print("=" * 60)
    print("  🧠 记忆系统演示")
    print("=" * 60)

    memory_tool = MemoryTool(user_id="demo_user")

    # 添加不同类型的记忆
    print("\n📝 添加记忆...")
    print(memory_tool.execute("add",
        content="用户张三是一名Python开发者，专注于机器学习和数据分析",
        memory_type="semantic", importance=0.8))
    
    print(memory_tool.execute("add",
        content="李四是前端工程师，擅长React和Vue.js开发",
        memory_type="semantic", importance=0.7))
    
    print(memory_tool.execute("add",
        content="今天讨论了Python的装饰器用法",
        memory_type="episodic", importance=0.6,
        event_type="discussion"))

    # 搜索记忆
    print("\n🔍 搜索 'Python开发者'...")
    print(memory_tool.execute("search", query="Python开发者"))

    print("\n🔍 搜索 '前端'...")
    print(memory_tool.execute("search", query="前端", limit=3))

    # 记忆统计
    print("\n📊 记忆统计:")
    print(memory_tool.execute("summary"))

    # 记忆整合
    print("\n🔄 整合记忆 (working → episodic)...")
    print(memory_tool.execute("consolidate", from_type="working", to_type="episodic"))

    # 遗忘不重要记忆
    print("\n🧹 遗忘不重要的记忆...")
    print(memory_tool.execute("forget", strategy="importance_based", threshold="0.2"))


def demo_rag():
    """演示RAG系统"""
    print("\n" + "=" * 60)
    print("  📚 RAG 知识检索演示")
    print("=" * 60)

    rag_tool = RAGTool(
        knowledge_base_path="./demo_knowledge_base",
        rag_namespace="demo"
    )

    # 添加知识
    print("\n📝 添加知识到知识库...")
    
    print(rag_tool.execute("add_text",
        text="Python是一种高级编程语言，由Guido van Rossum于1991年首次发布。"
             "Python的设计哲学强调代码的可读性和简洁的语法。",
        document_id="python_intro"))
    
    print(rag_tool.execute("add_text",
        text="机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式。"
             "主要包括监督学习、无监督学习和强化学习三种类型。",
        document_id="ml_basics"))
    
    print(rag_tool.execute("add_text",
        text="RAG（检索增强生成）是一种结合信息检索和文本生成的AI技术。"
             "它通过检索相关知识来增强大语言模型的生成能力，减少幻觉问题。",
        document_id="rag_concept"))

    # 搜索知识
    print("\n🔍 搜索 'Python编程语言'...")
    print(rag_tool.execute("search", query="Python编程语言的历史", limit=3))

    print("\n🔍 搜索 '机器学习'...")
    print(rag_tool.execute("search", query="机器学习的分支", limit=3))

    # 智能问答
    print("\n💡 智能问答: '什么是RAG?'...")
    print(rag_tool.execute("ask", question="什么是RAG技术？"))

    # 统计
    print("\n📊 知识库统计:")
    print(rag_tool.execute("stats"))


def demo_agent_with_memory():
    """演示Agent使用记忆和RAG工具"""
    print("\n" + "=" * 60)
    print("  🤖 Agent + 记忆 + RAG 综合演示")
    print("=" * 60)

    llm = NeoAgentLLM()
    memory_tool = MemoryTool(user_id="agent_demo")
    rag_tool = RAGTool(rag_namespace="agent_demo")

    registry = ToolRegistry()
    registry.register_tool(memory_tool)
    registry.register_tool(rag_tool)
    registry.register_tool(CalculatorTool())

    agent = SimpleAgent(
        name="智能助手",
        llm=llm,
        system_prompt="你是一个有记忆和知识检索能力的AI助手。在回答前，先检索记忆和知识库。",
        tool_registry=registry,
    )

    # 先添加一些知识
    rag_tool.execute("add_text",
        text="HelloAgents是一个教学友好的AI Agent框架，支持SimpleAgent、ReActAgent、"
             "ReflectionAgent和PlanAndSolveAgent四种智能体范式。",
        document_id="hello_agents")

    # 对话
    print("\n👤 用户: 请记住，我叫张三，正在学习AI Agent开发")
    response = agent.run("请记住，我叫张三，正在学习AI Agent开发")
    print(f"🤖 助手: {response}")


if __name__ == "__main__":
    demo_memory()
    demo_rag()
    # demo_agent_with_memory()  # 取消注释以体验Agent+记忆+RAG

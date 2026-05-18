"""ReflectionAgent 示例 - 自我反思与迭代优化（含记忆与RAG）"""

from dotenv import load_dotenv
load_dotenv()

from neo_agent_kit import NeoAgentLLM, ReflectionAgent, ToolRegistry
from neo_agent_kit.tools.builtin import MemoryTool, RAGTool


def main():
    llm = NeoAgentLLM()

    # ---- 通用反思模式 ----
    print("=" * 50)
    print("  1. 通用反思模式")
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
    print("  2. 代码审查模式")
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

    # ---- 反思 + 记忆记录 ----
    print("=" * 50)
    print("  3. 反思 + 记忆记录 🧠")
    print("=" * 50)

    memory_tool = MemoryTool(user_id="reflection_demo")

    # 记录每次反思过程
    memory_tool.execute("add",
        content="ReflectionAgent 通过「生成→反思→优化」循环提升输出质量",
        memory_type="semantic", importance=0.9, concept="agent_paradigm")

    article = agent.run("写一篇50字左右的短文介绍Python")
    memory_tool.execute("add",
        content=f"生成了关于Python的短文（经过反思优化）",
        memory_type="episodic", importance=0.6, event_type="generation")

    print(f"生成结果: {article}")
    print(f"\n记忆摘要: {memory_tool.execute('summary')}")

    # ---- 反思 + RAG 增强 ----
    print("\n" + "=" * 50)
    print("  4. 反思 + RAG 知识增强 📚")
    print("=" * 50)

    rag_tool = RAGTool(rag_namespace="reflection_demo")
    rag_tool.execute("add_text",
        text="ReflectionAgent 是 neo-agent-kit 四种智能体范式之一。"
             "它通过 initial → reflect → refine 三阶段循环提升输出质量。"
             "max_rounds 参数控制反思轮数，默认3轮。",
        document_id="reflection_info")

    # 使用RAG检索相关知识来辅助反思
    print(rag_tool.execute("search", query="ReflectionAgent 的工作原理"))
    print(rag_tool.execute("stats"))


if __name__ == "__main__":
    main()

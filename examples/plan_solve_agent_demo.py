"""PlanAndSolveAgent 示例 - 规划与执行分离（含记忆与RAG）"""

from dotenv import load_dotenv
load_dotenv()

from neo_agent_kit import NeoAgentLLM, PlanAndSolveAgent
from neo_agent_kit.tools.builtin import MemoryTool, RAGTool


def main():
    llm = NeoAgentLLM()

    # ---- 默认模式 ----
    print("=" * 50)
    print("  1. 默认规划执行模式")
    print("=" * 50)

    agent = PlanAndSolveAgent(name="规划助手", llm=llm)

    question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
    result = agent.run(question)
    print(f"\n📌 最终答案: {result}\n")

    # ---- 自定义提示词（数学专用）----
    print("=" * 50)
    print("  2. 数学专用模式")
    print("=" * 50)

    math_prompts = {
        "planner": (
            "你是数学问题规划专家。请将数学问题分解为计算步骤:\n\n"
            "问题: {question}\n\n"
            "输出格式:\n"
            '```python\n["计算步骤1", "计算步骤2", "求总和"]\n```\n'
        ),
        "executor": (
            "你是数学计算专家。请计算当前步骤:\n\n"
            "问题: {question}\n"
            "计划: {plan}\n"
            "历史: {history}\n"
            "当前步骤: {current_step}\n\n"
            "请只输出数值结果:"
        )
    }

    math_agent = PlanAndSolveAgent(
        name="数学专家",
        llm=llm,
        custom_prompts=math_prompts
    )

    result = math_agent.run(question)
    print(f"\n📌 最终答案: {result}\n")

    # ---- 规划 + 记忆记录 ----
    print("=" * 50)
    print("  3. 规划 + 记忆记录 🧠")
    print("=" * 50)

    memory_tool = MemoryTool(user_id="plan_demo")

    # 记录规划策略
    memory_tool.execute("add",
        content="PlanAndSolveAgent 将复杂问题先分解为子步骤，再逐步执行",
        memory_type="semantic", importance=0.9, concept="agent_paradigm")

    complex_question = "小明有20本书，小红有小明的一半，小刚比小红多3本。三人共有多少本书？"
    result = agent.run(complex_question)

    # 记录解题过程
    memory_tool.execute("add",
        content=f"解决了多步骤数学问题: {complex_question}",
        memory_type="episodic", importance=0.6, event_type="problem_solving")

    print(f"📌 最终答案: {result}")
    print(f"\n记忆摘要: {memory_tool.execute('summary')}")

    # ---- 规划 + RAG 知识增强 ----
    print("\n" + "=" * 50)
    print("  4. 规划 + RAG 知识增强 📚")
    print("=" * 50)

    rag_tool = RAGTool(rag_namespace="plan_demo")
    rag_tool.execute("add_text",
        text="PlanAndSolveAgent 先由 Planner 将问题分解为步骤列表，再由 Executor 逐步执行。"
             "每一步的历史结果会传递给后续步骤，最后汇总生成最终答案。",
        document_id="plan_solve_info")

    print(rag_tool.execute("search", query="PlanAndSolveAgent 的工作流程"))
    print(rag_tool.execute("stats"))


if __name__ == "__main__":
    main()

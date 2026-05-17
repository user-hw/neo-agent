"""PlanAndSolveAgent 示例 - 规划与执行分离"""

from dotenv import load_dotenv
load_dotenv()

from neo_agent_kit import NeoAgentLLM, PlanAndSolveAgent


def main():
    llm = NeoAgentLLM()

    # ---- 默认模式 ----
    print("=" * 50)
    print("  默认规划执行模式")
    print("=" * 50)

    agent = PlanAndSolveAgent(name="规划助手", llm=llm)

    question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
    result = agent.run(question)
    print(f"\n📌 最终答案: {result}\n")

    # ---- 自定义提示词（数学专用）----
    print("=" * 50)
    print("  数学专用模式")
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


if __name__ == "__main__":
    main()

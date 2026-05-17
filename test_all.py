"""
neo-agent-kit 完整测试文件

涵盖所有 Agent 范式和工具系统的测试。
运行方式:
    python test_all.py
"""

from dotenv import load_dotenv
load_dotenv()

from neo_agent_kit import (
    NeoAgentLLM, Config,
    SimpleAgent, ReActAgent, ReflectionAgent, PlanAndSolveAgent,
    ToolRegistry, Message,
)
from neo_agent_kit.tools.builtin import CalculatorTool


def print_separator(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ============================================================
# 测试 1: NeoAgentLLM 初始化
# ============================================================
def test_llm_init():
    print_separator("测试 1: NeoAgentLLM 初始化")

    try:
        llm = NeoAgentLLM()
        print(f"✅ Provider: {llm.provider}")
        print(f"✅ Model: {llm.model}")
        print(f"✅ Base URL: {llm.base_url}")
        return llm
    except Exception as e:
        print(f"⚠️ LLM 初始化跳过（需配置 API Key）: {e}")
        return None


# ============================================================
# 测试 2: 配置管理
# ============================================================
def test_config():
    print_separator("测试 2: 配置管理")

    config = Config.from_env()
    print(f"  Debug: {config.debug}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Max history: {config.max_history_length}")
    print(f"  Max react steps: {config.max_react_steps}")
    print(f"✅ 配置管理测试通过")

    return config


# ============================================================
# 测试 3: Message 系统
# ============================================================
def test_message():
    print_separator("测试 3: Message 系统")

    msg_user = Message("你好", "user")
    msg_assistant = Message("你好！有什么可以帮助你的？", "assistant")

    print(f"  User message: {msg_user}")
    print(f"  Assistant message: {msg_assistant}")
    print(f"  to_dict: {msg_user.to_dict()}")
    print(f"✅ Message 系统测试通过")


# ============================================================
# 测试 4: SimpleAgent
# ============================================================
def test_simple_agent(llm: NeoAgentLLM):
    print_separator("测试 4: SimpleAgent")

    agent = SimpleAgent(
        name="基础助手",
        llm=llm,
        system_prompt="你是一个友好的AI助手，请用简洁的方式回答问题。"
    )

    response = agent.run("你好，用一句话介绍你自己")
    print(f"  响应: {response[:200]}...")
    print(f"  历史消息数: {len(agent.get_history())}")
    print(f"✅ SimpleAgent 测试通过")
    return agent


# ============================================================
# 测试 5: SimpleAgent 流式响应
# ============================================================
def test_simple_agent_stream(llm: NeoAgentLLM):
    print_separator("测试 5: SimpleAgent 流式响应")

    agent = SimpleAgent(
        name="流式助手",
        llm=llm,
        system_prompt="你是一个友好的AI助手。"
    )

    print("  流式输出: ", end="")
    for chunk in agent.stream_run("什么是机器学习？用一句话回答"):
        pass  # 内容已在 stream_run 中实时打印
    print(f"\n  历史消息数: {len(agent.get_history())}")
    print(f"✅ 流式响应测试通过")


# ============================================================
# 测试 6: SimpleAgent 工具调用
# ============================================================
def test_simple_agent_tools(llm: NeoAgentLLM):
    print_separator("测试 6: SimpleAgent 工具调用")

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    agent = SimpleAgent(
        name="工具助手",
        llm=llm,
        system_prompt="你是一个智能助手，可以使用计算器工具帮助用户。当需要计算时请使用 [TOOL_CALL:calculator:表达式] 格式。",
        tool_registry=registry,
        enable_tool_calling=True
    )

    response = agent.run("请帮我计算 15 * 8 + 32")
    print(f"  响应: {response[:200]}...")
    print(f"✅ 工具调用测试通过")

    # 测试工具管理
    print(f"  可用工具: {agent.list_tools()}")
    print(f"  有工具: {agent.has_tools()}")


# ============================================================
# 测试 7: ReActAgent
# ============================================================
def test_react_agent(llm: NeoAgentLLM):
    print_separator("测试 7: ReActAgent")

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    agent = ReActAgent(
        name="ReAct助手",
        llm=llm,
        tool_registry=registry,
        max_steps=5
    )

    result = agent.run("计算 (10 + 5) * 2 - 8 的结果")
    print(f"  结果: {result[:200]}...")
    print(f"✅ ReActAgent 测试通过")


# ============================================================
# 测试 8: ReflectionAgent
# ============================================================
def test_reflection_agent(llm: NeoAgentLLM):
    print_separator("测试 8: ReflectionAgent")

    agent = ReflectionAgent(
        name="反思助手",
        llm=llm,
        max_rounds=2  # 控制轮数以节省 API 调用
    )

    result = agent.run("用一句话解释什么是深度学习")
    print(f"  最终结果: {result[:200]}...")
    print(f"✅ ReflectionAgent 测试通过")


# ============================================================
# 测试 9: PlanAndSolveAgent
# ============================================================
def test_plan_solve_agent(llm: NeoAgentLLM):
    print_separator("测试 9: PlanAndSolveAgent")

    agent = PlanAndSolveAgent(
        name="规划助手",
        llm=llm
    )

    question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
    result = agent.run(question)
    print(f"  最终结果: {result[:200]}...")
    print(f"✅ PlanAndSolveAgent 测试通过")


# ============================================================
# 测试 10: 工具注册表
# ============================================================
def test_tool_registry():
    print_separator("测试 10: 工具注册表")

    registry = ToolRegistry()

    # 注册 Tool 对象
    registry.register_tool(CalculatorTool())
    print(f"  工具数: {len(registry)}")
    print(f"  工具列表: {registry.list_tools()}")

    # 注册函数工具
    def my_echo(text: str) -> str:
        return f"Echo: {text}"

    registry.register_function("echo", "回显输入的文本", my_echo)
    print(f"  工具数: {len(registry)}")

    # 执行工具
    result = registry.execute_tool("calculator", "2 + 3 * 4")
    print(f"  计算器结果: {result}")

    result = registry.execute_tool("echo", "Hello World")
    print(f"  Echo 结果: {result}")

    # 工具描述
    print(f"  工具描述:\n{registry.get_tools_description()}")

    print(f"✅ 工具注册表测试通过")


# ============================================================
# 测试 11: 工具链
# ============================================================
def test_tool_chain():
    print_separator("测试 11: 工具链")

    from neo_agent_kit.tools.chain import ToolChain, ToolChainManager

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    def text_len(text: str) -> str:
        return f"文本长度: {len(text)}"

    registry.register_function("text_len", "计算文本长度", text_len)

    # 创建工具链
    chain = ToolChain("calc_and_measure", "计算并测量结果长度")
    chain.add_step("calculator", "{input}", "calc_result")
    chain.add_step("text_len", "{calc_result}", "length_result")

    manager = ToolChainManager(registry)
    manager.register_chain(chain)

    result = manager.execute_chain("calc_and_measure", "2 + 3 * 4")
    print(f"  工具链结果: {result}")
    print(f"✅ 工具链测试通过")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("  🧪 neo-agent-kit 完整测试套件")
    print("=" * 60)

    # 配置
    test_config()

    # Message
    test_message()

    # 工具注册表（不需要 LLM）
    test_tool_registry()

    # 工具链（不需要 LLM）
    test_tool_chain()

    # LLM 初始化
    llm = test_llm_init()
    if llm is None:
        print("\n⚠️ 跳过需要 LLM 的测试（请配置 .env 文件中的 API Key）")
        print("\n📋 快速配置:")
        print("  1. 复制 .env.example 为 .env")
        print("  2. 填入你的 API Key")
        print("  3. 重新运行测试")
        return

    # Agent 测试
    test_simple_agent(llm)
    test_simple_agent_stream(llm)
    test_simple_agent_tools(llm)
    test_react_agent(llm)
    test_reflection_agent(llm)
    test_plan_solve_agent(llm)

    print_separator("🎉 所有测试完成!")
    print("\n💡 提示:")
    print("  - SimpleAgent: 基础对话和工具调用")
    print("  - ReActAgent: 思考→行动→观察循环")
    print("  - ReflectionAgent: 执行→反思→优化循环")
    print("  - PlanAndSolveAgent: 先规划后执行")
    print("  - 工具系统: 注册、执行、链式调用")


if __name__ == "__main__":
    main()

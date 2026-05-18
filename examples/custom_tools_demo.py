"""自定义工具开发示例 - 展示如何扩展工具系统（含记忆与RAG集成）"""

from dotenv import load_dotenv
load_dotenv()

from neo_agent_kit import NeoAgentLLM, SimpleAgent, ToolRegistry
from neo_agent_kit.tools.base import Tool, ToolParameter
from neo_agent_kit.tools.chain import ToolChain, ToolChainManager
from neo_agent_kit.tools.builtin import MemoryTool, RAGTool
from typing import Dict, Any, List


# ========== 自定义工具1: 天气查询（模拟） ==========
class WeatherTool(Tool):
    """模拟天气查询工具"""

    def __init__(self):
        super().__init__(
            name="weather",
            description="查询指定城市的天气信息（模拟数据）"
        )
        self._weather_data = {
            "北京": "晴朗，25°C",
            "上海": "多云，28°C",
            "深圳": "阵雨，30°C",
            "杭州": "阴天，22°C",
        }

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="city",
                type="string",
                description="城市名称",
                required=True
            )
        ]

    def run(self, parameters: Dict[str, Any]) -> str:
        city = parameters.get("city", "")
        weather = self._weather_data.get(city, f"未找到 {city} 的天气数据")
        return f"🌤️ {city}: {weather}"


# ========== 自定义工具2: 翻译工具（模拟） ==========
class TranslateTool(Tool):
    """模拟翻译工具"""

    def __init__(self):
        super().__init__(
            name="translate",
            description="将文本翻译成指定语言（模拟）"
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="text", type="string", description="要翻译的文本", required=True),
            ToolParameter(name="target", type="string", description="目标语言", required=False, default="英文"),
        ]

    def run(self, parameters: Dict[str, Any]) -> str:
        text = parameters.get("text", "")
        target = parameters.get("target", "英文")
        mapping = {
            "你好": {"英文": "Hello", "日文": "こんにちは"},
            "谢谢": {"英文": "Thank you", "日文": "ありがとう"},
            "人工智能": {"英文": "Artificial Intelligence", "日文": "人工知能"},
        }
        translated = mapping.get(text, {}).get(target, f"[{text} 的{target}翻译]")
        return f"📝 翻译 ({target}): {translated}"


def demo_custom_tools():
    """演示自定义工具"""
    llm = NeoAgentLLM()

    # ---- 工具注册 ----
    registry = ToolRegistry()
    registry.register_tool(WeatherTool())
    registry.register_tool(TranslateTool())

    print("已注册工具:")
    print(registry.get_tools_description())

    # ---- 直接使用工具 ----
    print("\n" + "=" * 50)
    print("  1. 直接调用工具")
    print("=" * 50)

    print(f"天气: {registry.execute_tool('weather', '北京')}")
    print(f"翻译: {registry.execute_tool('translate', 'text=人工智能,target=日文')}")

    # ---- 工具链 ----
    print("\n" + "=" * 50)
    print("  2. 工具链调用")
    print("=" * 50)

    def echo(text: str) -> str:
        return text
    registry.register_function("echo", "回显输入", echo)

    chain = ToolChain("weather_report", "天气查询 + 翻译链")
    chain.add_step("weather", "{input}", "weather_info")
    chain.add_step("translate", "text={weather_info},target=日文", "translated")

    manager = ToolChainManager(registry)
    manager.register_chain(chain)

    result = manager.execute_chain("weather_report", "上海")
    print(f"工具链结果: {result}")

    # ---- Agent 集成 ----
    print("\n" + "=" * 50)
    print("  3. Agent 集成自定义工具")
    print("=" * 50)

    agent = SimpleAgent(
        name="全能助手",
        llm=llm,
        system_prompt="你可以使用天气查询和翻译工具。需要时用 [TOOL_CALL:工具名:参数] 格式调用。",
        tool_registry=registry,
        enable_tool_calling=True
    )

    print(f"Agent: {agent.run('北京今天天气怎么样？')}")


def demo_memory_rag_integration():
    """演示记忆与RAG工具集成到工具链"""
    print("\n" + "=" * 50)
    print("  4. 记忆 + RAG 工具集成 🧠📚")
    print("=" * 50)

    # 创建记忆和RAG工具
    memory_tool = MemoryTool(user_id="custom_demo")
    rag_tool = RAGTool(rag_namespace="custom_demo")

    # 添加知识到RAG
    rag_tool.execute("add_text",
        text="自定义工具需要继承 Tool 基类，实现 run 和 get_parameters 方法。"
             "ToolParameter 定义参数名、类型、描述和是否必填。"
             "to_openai_schema() 方法将工具转换为 OpenAI function calling 格式。",
        document_id="custom_tool_guide")

    # 记录到记忆
    memory_tool.execute("add",
        content="已学习如何创建自定义Tool：继承Tool基类，实现run和get_parameters",
        memory_type="semantic", importance=0.8)

    # 注册到工具注册表
    registry = ToolRegistry()
    registry.register_tool(memory_tool)
    registry.register_tool(rag_tool)
    registry.register_tool(WeatherTool())

    # 创建带记忆和RAG的工具链
    chain = ToolChain("weather_with_memory", "天气查询+记忆保存链")
    chain.add_step("weather", "{input}", "weather_result")
    chain.add_step("memory", "action=add,content=查询了{input}的天气：{weather_result},memory_type=episodic,importance=0.6", "saved")

    manager = ToolChainManager(registry)
    manager.register_chain(chain)

    result = manager.execute_chain("weather_with_memory", "杭州")
    print(f"工具链结果: {result}")

    # 查看记忆
    print(f"\n记忆检索: {memory_tool.execute('search', query='天气')}")

    # 查看RAG
    print(f"\nRAG搜索: {rag_tool.execute('search', query='自定义工具')}")
    print(f"RAG统计: {rag_tool.execute('stats')}")


def main():
    demo_custom_tools()
    demo_memory_rag_integration()


if __name__ == "__main__":
    main()

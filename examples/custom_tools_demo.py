"""自定义工具开发示例 - 展示如何扩展工具系统"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from neo_agent import NeoAgentLLM, SimpleAgent, ToolRegistry
from neo_agent.tools.base import Tool, ToolParameter
from neo_agent.tools.chain import ToolChain, ToolChainManager
from typing import Dict, Any, List


# ========== 自定义工具1: 天气查询（模拟） ==========
class WeatherTool(Tool):
    """模拟天气查询工具"""

    def __init__(self):
        super().__init__(
            name="weather",
            description="查询指定城市的天气信息（模拟数据）"
        )
        # 模拟天气数据
        self._weather_data = {
            "北京": "晴朗，25°C",
            "上海": "多云，28°C",
            "深圳": "阵雨，30°C",
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
        # 模拟翻译
        mapping = {
            "你好": {"英文": "Hello", "日文": "こんにちは"},
            "谢谢": {"英文": "Thank you", "日文": "ありがとう"},
        }
        translated = mapping.get(text, {}).get(target, f"[{text} 的{target}翻译]")
        return f"📝 翻译结果 ({target}): {translated}"


def main():
    llm = NeoAgentLLM()

    # ---- 工具注册 ----
    registry = ToolRegistry()
    registry.register_tool(WeatherTool())
    registry.register_tool(TranslateTool())

    print("已注册工具:")
    print(registry.get_tools_description())

    # ---- 直接使用工具 ----
    print("\n" + "=" * 50)
    print("  直接调用工具")
    print("=" * 50)

    weather_result = registry.execute_tool("weather", "北京")
    print(f"天气查询: {weather_result}")

    translate_result = registry.execute_tool("translate", "text=你好,target=日文")
    print(f"翻译结果: {translate_result}")

    # ---- 工具链 ----
    print("\n" + "=" * 50)
    print("  工具链调用")
    print("=" * 50)

    # 注册一个 echo 函数
    def echo(text: str) -> str:
        return text

    registry.register_function("echo", "回显输入", echo)

    chain = ToolChain("weather_report", "天气查询报告链")
    chain.add_step("weather", "{input}", "weather_info")
    chain.add_step("translate", "text={weather_info},target=日文", "translated")

    manager = ToolChainManager(registry)
    manager.register_chain(chain)

    result = manager.execute_chain("weather_report", "北京")
    print(f"工具链最终结果: {result}")

    # ---- 与 Agent 集成 ----
    print("\n" + "=" * 50)
    print("  Agent 集成")
    print("=" * 50)

    agent = SimpleAgent(
        name="全能助手",
        llm=llm,
        system_prompt="你可以使用天气查询和翻译工具。需要时用 [TOOL_CALL:工具名:参数] 格式调用。",
        tool_registry=registry,
        enable_tool_calling=True
    )

    response = agent.run("北京今天天气怎么样？")
    print(f"Agent 响应: {response}")


if __name__ == "__main__":
    main()

"""多源搜索工具"""
import os
from typing import Dict, Any, List, Optional
from ..base import Tool, ToolParameter


class SearchTool(Tool):
    """智能多源搜索工具

    支持多种搜索后端，按优先级自动选择:
    1. Tavily API (AI 优化的搜索)
    2. SerpApi (Google 搜索)

    降级策略: Tavily → SerpApi → 错误提示
    """

    def __init__(
        self,
        backend: str = "hybrid",
        tavily_key: Optional[str] = None,
        serpapi_key: Optional[str] = None
    ):
        super().__init__(
            name="search",
            description="智能网页搜索引擎，支持混合搜索模式，自动选择最佳搜索源。"
        )
        self.backend = backend
        self.tavily_key = tavily_key or os.getenv("TAVILY_API_KEY")
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")
        self.available_backends = []
        self._setup_backends()

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="搜索查询字符串",
                required=True
            )
        ]

    def _setup_backends(self):
        """检测并设置可用的搜索后端"""
        # 检测 Tavily
        if self.tavily_key:
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=self.tavily_key)
                self.available_backends.append("tavily")
                print("✅ Tavily 搜索源已启用")
            except ImportError:
                print("⚠️ Tavily 库未安装，请执行: pip install tavily-python")
            except Exception as e:
                print(f"⚠️ Tavily 初始化失败: {e}")

        # 检测 SerpApi
        if self.serpapi_key:
            self.available_backends.append("serpapi")
            print("✅ SerpApi 搜索源已启用")

        if self.available_backends:
            print(f"🔧 可用搜索源: {', '.join(self.available_backends)}")
        else:
            print("⚠️ 没有可用的搜索 API，请设置 TAVILY_API_KEY 或 SERPAPI_API_KEY")

    def run(self, parameters: Dict[str, Any]) -> str:
        query = parameters.get("query", "")
        if not query.strip():
            return "❌ 错误: 搜索查询不能为空"

        if self.backend == "hybrid":
            return self._search_hybrid(query)
        elif self.backend == "tavily":
            return self._search_tavily(query)
        elif self.backend == "serpapi":
            return self._search_serpapi(query)
        else:
            return f"❌ 不支持的搜索后端: {self.backend}"

    def _search_hybrid(self, query: str) -> str:
        """混合搜索 - 智能选择最佳搜索源"""
        if "tavily" in self.available_backends:
            try:
                return self._search_tavily(query)
            except Exception as e:
                print(f"⚠️ Tavily 搜索失败: {e}")
                if "serpapi" in self.available_backends:
                    print("🔄 切换到 SerpApi 搜索")
                    return self._search_serpapi(query)

        if "serpapi" in self.available_backends:
            try:
                return self._search_serpapi(query)
            except Exception as e:
                print(f"⚠️ SerpApi 搜索失败: {e}")

        return (
            "❌ 没有可用的搜索源，请配置以下 API 密钥之一:\n\n"
            "1. Tavily API: 设置环境变量 TAVILY_API_KEY\n"
            "   获取地址: https://tavily.com/\n\n"
            "2. SerpAPI: 设置环境变量 SERPAPI_API_KEY\n"
            "   获取地址: https://serpapi.com/\n\n"
            "配置后重新运行程序。"
        )

    def _search_tavily(self, query: str) -> str:
        """使用 Tavily 搜索"""
        try:
            response = self.tavily_client.search(
                query=query,
                search_depth="basic",
                include_answer=True,
                max_results=3
            )

            result = ""
            if answer := response.get('answer'):
                result += f"💡 AI 直接答案: {answer}\n\n"

            result += "🔗 相关结果:\n"
            for i, item in enumerate(response.get('results', [])[:3], 1):
                result += f"[{i}] {item.get('title', '')}\n"
                content = item.get('content', '')[:200]
                result += f"    {content}...\n"
                result += f"    来源: {item.get('url', '')}\n\n"

            return result
        except Exception as e:
            return f"❌ Tavily 搜索失败: {e}"

    def _search_serpapi(self, query: str) -> str:
        """使用 SerpApi 搜索"""
        try:
            from serpapi import GoogleSearch

            search = GoogleSearch({
                "q": query,
                "api_key": self.serpapi_key,
                "num": 3
            })
            results = search.get_dict()

            result = "🌐 Google 搜索结果:\n"
            if "organic_results" in results:
                for i, res in enumerate(results["organic_results"][:3], 1):
                    result += f"[{i}] {res.get('title', '')}\n"
                    result += f"    {res.get('snippet', '')}\n"
                    result += f"    来源: {res.get('link', '')}\n\n"
            return result
        except ImportError:
            return "❌ SerpApi 库未安装，请执行: pip install google-search-results"
        except Exception as e:
            return f"❌ SerpApi 搜索失败: {e}"

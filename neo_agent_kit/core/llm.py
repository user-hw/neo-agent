"""NeoAgentLLM - 统一大语言模型接口

支持多提供商、本地模型、自动检测机制。
"""
import os
from typing import Optional, List, Dict, Any, Iterator
from openai import OpenAI

from .exceptions import LLMError, ConfigError
from .config import Config


class NeoAgentLLM:
    """统一的大语言模型调用中枢

    支持提供商:
        - openai: OpenAI 官方 API
        - modelscope: ModelScope (魔搭社区)
        - zhipu: 智谱 AI (GLM 系列)
        - deepseek: DeepSeek
        - ollama: 本地 Ollama 部署
        - vllm: 本地 VLLM 部署
        - auto: 自动检测
    """

    # 各提供商的默认配置
    PROVIDER_CONFIGS = {
        "openai": {
            "env_key": "OPENAI_API_KEY",
            "default_base_url": "https://api.openai.com/v1",
            "default_model": "gpt-3.5-turbo",
        },
        "modelscope": {
            "env_key": "MODELSCOPE_API_KEY",
            "default_base_url": "https://api-inference.modelscope.cn/v1/",
            "default_model": "Qwen/Qwen2.5-VL-72B-Instruct",
        },
        "zhipu": {
            "env_key": "ZHIPU_API_KEY",
            "default_base_url": "https://open.bigmodel.cn/api/paas/v4/",
            "default_model": "glm-4-flash",
        },
        "deepseek": {
            "env_key": "DEEPSEEK_API_KEY",
            "default_base_url": "https://api.deepseek.com/v1",
            "default_model": "deepseek-chat",
        },
        "ollama": {
            "env_key": None,
            "default_base_url": "http://localhost:11434/v1",
            "default_model": "llama3",
        },
        "vllm": {
            "env_key": None,
            "default_base_url": "http://localhost:8000/v1",
            "default_model": "Qwen/Qwen1.5-0.5B-Chat",
        },
    }

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
        **kwargs
    ):
        """初始化 NeoAgentLLM

        Args:
            model: 模型名称
            api_key: API 密钥
            base_url: API 基础地址
            provider: 提供商 (openai/modelscope/zhipu/deepseek/ollama/vllm/auto)
            temperature: 温度参数
            max_tokens: 最大 token 数
            timeout: 请求超时时间
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_kwargs = kwargs

        # 自动检测或使用指定的 provider
        if provider == "auto":
            self.provider = self._auto_detect_provider(api_key, base_url)
        else:
            self.provider = provider

        # 解析凭证
        self.api_key, self.base_url = self._resolve_credentials(api_key, base_url)

        # 设置模型
        if model:
            self.model = model
        elif provider_config := self.PROVIDER_CONFIGS.get(self.provider):
            self.model = os.getenv("LLM_MODEL_ID") or provider_config["default_model"]
        else:
            self.model = os.getenv("LLM_MODEL_ID") or "gpt-3.5-turbo"

        # 创建 OpenAI 客户端
        if not self.api_key:
            raise ConfigError(
                f"未找到 {self.provider} 的 API Key。"
                f"请设置环境变量或通过 api_key 参数传入。"
            )
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

        print(f"✅ NeoAgentLLM 初始化完成: provider={self.provider}, model={self.model}")

    # ========== 自动检测机制 ==========

    def _auto_detect_provider(self, api_key: Optional[str], base_url: Optional[str]) -> str:
        """自动检测 LLM 提供商

        优先级:
        1. 检查特定提供商的环境变量
        2. 根据 base_url 判断
        3. 根据 API 密钥格式判断
        4. 默认返回 'auto'（使用通用配置）
        """
        # 1. 检查特定提供商环境变量（最高优先级）
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("MODELSCOPE_API_KEY"):
            return "modelscope"
        if os.getenv("ZHIPU_API_KEY"):
            return "zhipu"
        if os.getenv("DEEPSEEK_API_KEY"):
            return "deepseek"

        # 获取通用环境变量
        actual_api_key = api_key or os.getenv("LLM_API_KEY")
        actual_base_url = base_url or os.getenv("LLM_BASE_URL")

        # 2. 根据 base_url 判断
        if actual_base_url:
            base_url_lower = actual_base_url.lower()
            # 域名匹配
            if "api.openai.com" in base_url_lower:
                return "openai"
            if "api-inference.modelscope.cn" in base_url_lower:
                return "modelscope"
            if "open.bigmodel.cn" in base_url_lower:
                return "zhipu"
            if "api.deepseek.com" in base_url_lower:
                return "deepseek"
            # 端口匹配（本地服务）
            if "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
                if ":11434" in base_url_lower:
                    return "ollama"
                if ":8000" in base_url_lower:
                    return "vllm"
                return "local"

        # 3. 根据 API 密钥格式辅助判断
        if actual_api_key:
            if actual_api_key.startswith("ms-"):
                return "modelscope"
            if actual_api_key.startswith("sk-"):
                return "openai"

        # 4. 默认返回 auto，使用通用配置
        return "auto"

    def _resolve_credentials(self, api_key: Optional[str], base_url: Optional[str]) -> tuple:
        """根据 provider 解析 API 密钥和 base_url"""
        provider_config = self.PROVIDER_CONFIGS.get(self.provider, {})

        # 解析 API Key
        env_key = provider_config.get("env_key")
        if env_key:
            resolved_api_key = api_key or os.getenv(env_key) or os.getenv("LLM_API_KEY")
        else:
            # 本地服务通常不需要真实 API Key
            resolved_api_key = api_key or os.getenv("LLM_API_KEY") or self.provider

        # 解析 base_url
        resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or provider_config.get(
            "default_base_url", "https://api.openai.com/v1"
        )

        return resolved_api_key, resolved_base_url

    # ========== 核心调用方法 ==========

    def think(self, messages: List[Dict[str, Any]], **kwargs) -> Iterator[str]:
        """流式调用 LLM，逐块返回响应

        Args:
            messages: 消息列表

        Yields:
            响应文本的每个 chunk
        """
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise LLMError(f"LLM 流式调用失败: {e}") from e

    def invoke(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """同步调用 LLM，返回完整响应

        Args:
            messages: 消息列表

        Returns:
            完整的响应文本
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                stream=False,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise LLMError(f"LLM 调用失败: {e}") from e

    def stream_invoke(self, messages: List[Dict[str, Any]], **kwargs) -> Iterator[str]:
        """流式调用 LLM（think 的别名，保持接口兼容）"""
        return self.think(messages, **kwargs)

    def invoke_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        **kwargs
    ):
        """使用 OpenAI Function Calling 原生能力调用 LLM

        Args:
            messages: 消息列表
            tools: 工具 schema 列表（OpenAI 格式）
            tool_choice: 工具选择策略

        Returns:
            OpenAI API 原始响应对象
        """
        try:
            return self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
            )
        except Exception as e:
            raise LLMError(f"LLM 函数调用失败: {e}") from e

    def __repr__(self) -> str:
        return f"NeoAgentLLM(provider={self.provider}, model={self.model})"

# neo-agent 🚀

一个轻量级、教学友好的 AI Agent 框架。从零开始构建，帮助理解 Agent 的底层工作原理。

## ✨ 特性

- **🎯 轻量级**: 核心依赖仅 `openai` + `pydantic` + `python-dotenv`
- **🔌 多提供商**: 支持 OpenAI / ModelScope / 智谱 / DeepSeek / Ollama / VLLM
- **🧠 四种 Agent 范式**: SimpleAgent / ReActAgent / ReflectionAgent / PlanAndSolveAgent
- **🛠️ 万物皆工具**: 统一工具抽象，内置计算器和多源搜索工具
- **📖 教学友好**: 代码清晰注释，渐进式学习路径
- **🔍 自动检测**: 智能推断 LLM 提供商，零配置即可运行

## 📦 安装

```bash
pip install -e .

# 带搜索工具支持
pip install -e ".[search]"
```

## 🚀 快速开始

### 1. 配置环境变量

创建 `.env` 文件:

```bash
# OpenAI (默认)
OPENAI_API_KEY="sk-your-api-key"

# 或其他提供商
# MODELSCOPE_API_KEY="your-key"
# ZHIPU_API_KEY="your-key"
# DEEPSEEK_API_KEY="your-key"

# 本地模型
# LLM_BASE_URL="http://localhost:11434/v1"  # Ollama
# LLM_BASE_URL="http://localhost:8000/v1"   # VLLM
```

### 2. 基础对话

```python
from dotenv import load_dotenv
from neo_agent import NeoAgentLLM, SimpleAgent

load_dotenv()

llm = NeoAgentLLM()  # 自动检测 provider
agent = SimpleAgent(name="AI助手", llm=llm, system_prompt="你是一个有用的AI助手")

response = agent.run("你好！请介绍一下自己")
print(response)
```

### 3. ReAct Agent（工具调用）

```python
from neo_agent import NeoAgentLLM, ReActAgent, ToolRegistry
from neo_agent.tools.builtin import CalculatorTool

llm = NeoAgentLLM()
registry = ToolRegistry()
registry.register_tool(CalculatorTool())

agent = ReActAgent(name="计算助手", llm=llm, tool_registry=registry)
result = agent.run("计算 (15 + 7) * 3 的结果")
print(result)
```

### 4. ReflectionAgent（反思优化）

```python
from neo_agent import NeoAgentLLM, ReflectionAgent

llm = NeoAgentLLM()
agent = ReflectionAgent(name="写作助手", llm=llm)

result = agent.run("写一篇关于人工智能发展历程的简短文章")
print(result)
```

## 🏗️ 架构

```
neo_agent/
├── core/                    # 核心层
│   ├── agent.py            # Agent 抽象基类
│   ├── llm.py              # NeoAgentLLM 统一接口
│   ├── message.py          # 消息系统
│   ├── config.py           # 配置管理
│   └── exceptions.py       # 异常体系
├── agents/                  # Agent 实现层
│   ├── simple_agent.py     # SimpleAgent
│   ├── react_agent.py      # ReActAgent
│   ├── reflection_agent.py # ReflectionAgent
│   └── plan_solve_agent.py # PlanAndSolveAgent
└── tools/                   # 工具系统层
    ├── base.py             # 工具基类
    ├── registry.py         # 工具注册机制
    ├── chain.py            # 工具链管理
    ├── async_executor.py   # 异步执行器
    └── builtin/            # 内置工具
        ├── calculator.py   # 计算工具
        └── search.py       # 搜索工具
```

## 📄 License

MIT

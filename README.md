# neo-agent-kit 🚀

一个轻量级、教学友好的 AI Agent 框架。从零开始构建，帮助理解 Agent 的底层工作原理。

## ✨ 特性

- **🎯 轻量级**: 核心依赖仅 `openai` + `pydantic` + `python-dotenv`
- **🔌 多提供商**: 支持 OpenAI / ModelScope / 智谱 / DeepSeek / Ollama / VLLM
- **🧠 四种 Agent 范式**: SimpleAgent / ReActAgent / ReflectionAgent / PlanAndSolveAgent
- **� 记忆系统**: 工作记忆 / 情景记忆 / 语义记忆，支持整合与遗忘机制
- **📚 RAG 系统**: 多格式文档处理 + 智能分块 + 检索增强生成
- **🛠️ 万物皆工具**: 统一工具抽象，内置计算器、搜索、记忆、RAG 工具
- **📖 教学友好**: 代码清晰注释，渐进式学习路径
- **🔍 自动检测**: 智能推断 LLM 提供商，零配置即可运行

## 📦 安装

```bash
pip install neo-agent-kit
```

带可选依赖：

```bash
# 包含搜索 + 记忆 + RAG 全部功能
pip install "neo-agent-kit[all]"

# 仅搜索工具
pip install "neo-agent-kit[search]"

# 仅记忆系统（含本地嵌入模型）
pip install "neo-agent-kit[memory]"

# 仅 RAG（含 PDF/Word 支持）
pip install "neo-agent-kit[rag]"
```

> 💡 如需安装最新开发版：`pip install git+https://github.com/user-hw/neo-agent.git`

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
from neo_agent_kit import NeoAgentLLM, SimpleAgent

load_dotenv()

llm = NeoAgentLLM()  # 自动检测 provider
agent = SimpleAgent(name="AI助手", llm=llm, system_prompt="你是一个有用的AI助手")

response = agent.run("你好！请介绍一下自己")
print(response)
```

### 3. 记忆系统

```python
from dotenv import load_dotenv
from neo_agent_kit import NeoAgentLLM, SimpleAgent, ToolRegistry
from neo_agent_kit.tools.builtin import MemoryTool

load_dotenv()

llm = NeoAgentLLM()
memory_tool = MemoryTool(user_id="user123")

registry = ToolRegistry()
registry.register_tool(memory_tool)

agent = SimpleAgent(name="记忆助手", llm=llm, tool_registry=registry)

# 添加记忆
print(memory_tool.execute("add", content="用户张三是一名Python开发者", memory_type="semantic", importance=0.8))

# 搜索记忆
print(memory_tool.execute("search", query="Python开发者"))

# 记忆整合（短期 → 长期）
print(memory_tool.execute("consolidate", from_type="working", to_type="episodic"))
```

### 4. RAG 知识检索

```python
from neo_agent_kit.tools.builtin import RAGTool

rag_tool = RAGTool(knowledge_base_path="./knowledge_base")

# 添加文本
print(rag_tool.execute("add_text", text="Python 由 Guido van Rossum 于1991年发布，强调代码可读性。"))

# 搜索
print(rag_tool.execute("search", query="Python 的历史"))

# 添加 PDF 文件
print(rag_tool.execute("add_document", file_path="./docs/guide.pdf"))
```

### 5. 命令行使用

```bash
neo-agent-kit ask "你好，请用一句话介绍一下你自己"
python -m neo_agent_kit ask "帮我总结一下 ReAct Agent 是什么"
```

## 🏗️ 架构

```
neo_agent_kit/
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
├── memory/                  # 记忆系统 (v0.2.0 新增)
│   ├── base.py             # MemoryItem / MemoryConfig / BaseMemory
│   ├── manager.py          # MemoryManager 统一协调
│   ├── embedding.py        # 嵌入服务 (TF-IDF / Local / DashScope)
│   └── types/              # 记忆类型实现
│       ├── working.py      # 工作记忆 (TTL, 纯内存)
│       ├── episodic.py     # 情景记忆 (SQLite 持久化)
│       └── semantic.py     # 语义记忆 (概念 + 实体提取)
├── rag/                     # RAG 系统 (v0.2.0 新增)
│   ├── document.py         # 文档处理 (多格式 + 智能分块)
│   └── pipeline.py         # RAG 管道 (索引/检索/生成)
└── tools/                   # 工具系统层
    ├── base.py             # 工具基类
    ├── registry.py         # 工具注册机制
    ├── chain.py            # 工具链管理
    ├── async_executor.py   # 异步执行器
    └── builtin/            # 内置工具
        ├── calculator.py   # 计算工具
        ├── search.py       # 搜索工具
        ├── memory_tool.py  # 记忆工具 (v0.2.0 新增)
        └── rag_tool.py     # RAG 工具 (v0.2.0 新增)
```

## 📄 License

MIT

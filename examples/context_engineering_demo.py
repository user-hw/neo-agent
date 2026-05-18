"""
上下文工程快速体验 - ContextBuilder + NoteTool + TerminalTool

演示 GSSC (Gather-Select-Structure-Compress) 流水线、
结构化笔记和安全的终端操作。
"""
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
from neo_agent_kit import (
    NeoAgentLLM, SimpleAgent, ToolRegistry,
    ContextBuilder, ContextConfig, ContextPacket,
)
from neo_agent_kit.tools.builtin import (
    MemoryTool, RAGTool, NoteTool, TerminalTool, CalculatorTool,
)


def demo_context_builder():
    """演示 GSSC 上下文构建流水线"""
    print("=" * 60)
    print("  1. ContextBuilder: GSSC 流水线")
    print("=" * 60)

    # 准备数据源
    memory_tool = MemoryTool(user_id="ctx_demo")
    rag_tool = RAGTool(rag_namespace="ctx_demo")

    # 添加一些知识和记忆
    rag_tool.execute("add_text",
        text="上下文工程关注在每次LLM调用前，优化输入的token集合。"
             "GSSC流水线包括Gather(汇集)、Select(选择)、Structure(结构化)、Compress(压缩)四个阶段。",
        document_id="ctx_intro")
    rag_tool.execute("add_text",
        text="ContextBuilder的设计理念：上下文是有限资源，具有边际收益递减。"
             "应追求高信号密度的信息，避免上下文腐蚀(rot)。",
        document_id="ctx_principles")

    memory_tool.execute("add",
        content="用户正在学习上下文工程，已掌握Memory和RAG基础",
        memory_type="semantic", importance=0.8)

    # 创建 ContextBuilder
    builder = ContextBuilder(
        memory_tool=memory_tool,
        rag_tool=rag_tool,
        config=ContextConfig(
            max_tokens=3000,
            reserve_ratio=0.2,
            min_relevance=0.1,
            enable_compression=True,
            recency_weight=0.3,
            relevance_weight=0.7,
        ),
    )

    # 模拟对话历史
    from neo_agent_kit import Message
    history = [
        Message(content="我正在学习AI Agent开发", role="user"),
        Message(content="很好！我们从基础开始...", role="assistant"),
        Message(content="什么是上下文工程？", role="user"),
    ]

    # 构建上下文
    context = builder.build(
        user_query="什么是上下文工程？和提示工程有什么区别？",
        conversation_history=history,
        system_instructions="你是一位AI工程专家，回答要准确、有据。",
    )

    print(context)
    print()

    # 演示自定义 ContextPacket
    packets = [
        ContextPacket(
            content="[代码库结构]\n src/models/  - 数据模型\n src/services/ - 业务逻辑",
            timestamp=datetime.now(),
            token_count=50,
            relevance_score=0.8,
            metadata={"type": "code_structure"},
        ),
    ]

    context2 = builder.build(
        user_query="如何重构数据模型层？",
        custom_packets=packets,
        conversation_history=history,
        system_instructions="你是代码重构专家。",
    )
    print(context2[:500] + "...\n")


def demo_note_tool():
    """演示 NoteTool 结构化笔记"""
    print("=" * 60)
    print("  2. NoteTool: 结构化笔记 📝")
    print("=" * 60)

    notes = NoteTool(workspace="./demo_notes")

    # 创建不同类型的笔记
    print("\n📝 创建笔记...")
    print(notes.execute("create",
        title="重构项目 - 第一阶段",
        content="## 完成情况\n已完成数据模型层重构，测试覆盖率85%。\n\n## 下一步\n重构业务逻辑层。",
        note_type="task_state",
        tags="refactoring,phase1"))

    print(notes.execute("create",
        title="依赖冲突问题",
        content="## 问题\n第三方库版本不兼容。\n\n## 影响\n业务逻辑层3个模块受影响。",
        note_type="blocker",
        tags="dependency,urgent"))

    print(notes.execute("create",
        title="性能优化建议",
        content="## 建议\n1. 使用缓存减少DB查询\n2. 引入异步处理\n\n## 预期提升\n响应时间减少40%",
        note_type="action",
        tags="performance,optimization"))

    # 搜索笔记
    print("\n🔍 搜索 '重构'...")
    print(notes.execute("search", query="重构"))

    print("\n🔍 搜索 '依赖'...")
    print(notes.execute("search", query="依赖"))

    # 按类型过滤
    print("\n📋 列出所有 blocker...")
    print(notes.execute("list", note_type="blocker"))

    # 摘要
    print("\n📊 笔记摘要:")
    print(notes.execute("summary"))

    # 读取单个笔记
    print("\n📄 读取笔记...")
    # 从摘要中获取第一个笔记ID
    import os, json
    idx_file = "./demo_notes/notes_index.json"
    if os.path.exists(idx_file):
        with open(idx_file) as f:
            idx = json.load(f)
        if idx:
            first_id = list(idx.keys())[0]
            print(notes.execute("read", note_id=first_id))


def demo_terminal_tool():
    """演示 TerminalTool 安全终端操作"""
    print("\n" + "=" * 60)
    print("  3. TerminalTool: 即时文件探索 💻")
    print("=" * 60)

    # 在当前目录操作
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建测试文件
        os.makedirs(os.path.join(tmpdir, "src"), exist_ok=True)
        with open(os.path.join(tmpdir, "README.md"), "w") as f:
            f.write("# Test Project\n\nThis is a demo project.\n\n## TODO\n- Add more features\n- Fix bug #42")
        with open(os.path.join(tmpdir, "src", "main.py"), "w") as f:
            f.write("def main():\n    print('Hello World')\n\nif __name__ == '__main__':\n    main()\n")

        term = TerminalTool(workspace=tmpdir, timeout=10)

        # 列出文件
        print("\n📁 文件列表:")
        print(term.execute("ls -la"))

        # 查看内容
        print("\n📄 README.md:")
        print(term.execute("cat README.md"))

        # 搜索 TODO
        print("\n🔍 搜索 TODO:")
        print(term.execute("grep -r 'TODO' ."))

        # 统计行数
        print("\n📊 代码统计:")
        print(term.execute("find . -name '*.py' -exec wc -l {} +"))

        # 导航
        print("\n📂 目录导航:")
        print(term.execute("cd src"))
        print(term.execute("pwd"))
        print(term.execute("ls"))
        term.reset_dir()  # 回到工作目录

        # 安全检查演示
        print("\n🛡️ 安全检查:")
        print(term.execute("cat /etc/passwd"))  # 应该被拦截


def demo_integration():
    """演示 ContextBuilder + NoteTool + TerminalTool 协同"""
    print("\n" + "=" * 60)
    print("  4. 三者协同: 代码库探索助手 🔗")
    print("=" * 60)

    import tempfile, os, json

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建模拟项目
        os.makedirs(os.path.join(tmpdir, "app", "models"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "app", "services"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "tests"), exist_ok=True)

        with open(os.path.join(tmpdir, "app", "models", "user.py"), "w") as f:
            f.write("class User:\n    def __init__(self, name, email):\n        self.name = name\n        self.email = email\n")

        with open(os.path.join(tmpdir, "app", "services", "user_service.py"), "w") as f:
            f.write("# TODO: add validation\n# FIXME: email format check needed\n\ndef create_user(name, email):\n    return {'name': name, 'email': email}\n")

        # 初始化工具
        term = TerminalTool(workspace=tmpdir)
        notes = NoteTool(workspace="./demo_integration_notes")
        memory_tool = MemoryTool(user_id="integrated_demo")

        # 1. 用 TerminalTool 探索项目
        print("\n🔍 探索项目结构...")
        structure = term.execute("find . -type f -name '*.py'")
        print(structure)

        todos = term.execute("grep -rn 'TODO\\|FIXME' --include='*.py' .")
        print(f"\n📋 待办事项:\n{todos}")

        # 2. 用 NoteTool 记录发现
        print("\n📝 记录发现到笔记...")
        print(notes.execute("create",
            title="代码库探索发现",
            content=f"## 项目结构\n```\n{structure}\n```\n\n## 待办事项\n```\n{todos}\n```",
            note_type="task_state",
            tags="exploration,initial"))

        # 3. 用 ContextBuilder 构建上下文
        from neo_agent_kit import ContextBuilder, ContextConfig, ContextPacket

        packets = [
            ContextPacket(
                content=f"[代码结构]\n{structure}",
                timestamp=datetime.now(),
                token_count=len(structure) // 4,
                relevance_score=0.8,
                metadata={"type": "code_structure"},
            ),
        ]

        builder = ContextBuilder(
            memory_tool=memory_tool,
            config=ContextConfig(max_tokens=2000),
        )

        context = builder.build(
            user_query="检查这个项目的代码质量",
            system_instructions="你是代码审查专家。基于代码结构给出建议。",
            custom_packets=packets,
        )

        print(f"\n⚙️ 构建的上下文 ({len(context)} 字符):")
        print(context[:500] + "...")


def main():
    print("🚀 neo-agent-kit 上下文工程体验\n")
    demo_context_builder()
    demo_note_tool()
    demo_terminal_tool()
    demo_integration()

    # 清理
    import shutil, os
    for d in ["./demo_notes", "./demo_integration_notes", "./memory_data"]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
    print("\n✅ 体验完成！")


if __name__ == "__main__":
    main()

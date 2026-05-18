"""终端工具 (TerminalTool)

为智能体提供安全的命令行执行能力，支持即时文件系统探索。

安全机制 (四层防护):
1. 命令白名单: 仅允许只读/安全的命令
2. 工作目录沙箱: 限制在工作目录内
3. 超时控制: 防止无限循环
4. 输出大小限制: 防止内存溢出
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ..base import Tool, ToolParameter


# 允许的安全命令白名单（只读 + 安全）
ALLOWED_COMMANDS = {
    # 文件列表与信息
    "ls", "dir", "tree",
    # 文件内容查看
    "cat", "head", "tail",
    # 文件搜索
    "find", "grep", "egrep", "fgrep",
    # 文本处理
    "wc", "sort", "uniq", "cut", "awk", "sed",
    # 目录操作
    "pwd", "cd",
    # 文件信息
    "file", "stat", "du", "df",
    # 进程信息 (只读)
    "ps", "top",
    # 其他安全命令
    "echo", "which", "whereis", "env", "printenv",
    # Git (只读操作)
    "git",
}


class TerminalTool(Tool):
    """终端工具 - 安全的命令行执行

    使用示例:
        term = TerminalTool(workspace="./my_project")
        term.run({"command": "ls -la"})
        term.run({"command": "grep -r 'TODO' --include='*.py' ."})
        term.run({"command": "cat README.md"})
    """

    def __init__(
        self,
        workspace: str = ".",
        timeout: int = 30,
        max_output_size: int = 500_000,  # 500KB
        allow_cd: bool = True,
    ):
        super().__init__(
            name="terminal",
            description=(
                "安全的终端命令执行工具。支持文件浏览(ls, tree)、内容查看(cat, head, tail)、"
                "文本搜索(grep, find)、统计(wc, du)等只读操作。"
            ),
        )
        self.workspace = Path(workspace).resolve()
        self.current_dir = self.workspace
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.allow_cd = allow_cd

        if not self.workspace.exists():
            os.makedirs(self.workspace, exist_ok=True)

        print(f"💻 TerminalTool 初始化: workspace={self.workspace}, timeout={timeout}s")

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                type="string",
                description="要执行的命令 (仅允许安全的只读命令)",
                required=True,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> str:
        command = parameters.get("command", "").strip()
        if not command:
            return "❌ 命令不能为空"

        return self._execute(command)

    def execute(self, action: str = "", **kwargs) -> str:
        """便捷执行"""
        kwargs.setdefault("command", action)
        return self.run(kwargs)

    # ========== 核心 ==========

    def _execute(self, command: str) -> str:
        """执行命令（带安全检查）"""
        # 1. 解析命令名
        cmd_parts = command.split()
        if not cmd_parts:
            return "❌ 命令不能为空"

        cmd_name = cmd_parts[0]

        # 2. 白名单检查
        if cmd_name == "cd":
            return self._handle_cd(cmd_parts)

        if cmd_name not in ALLOWED_COMMANDS:
            allowed = ", ".join(sorted(ALLOWED_COMMANDS))
            return f"❌ 不允许的命令: {cmd_name}\n允许的命令: {allowed}"

        # 3. 路径安全检查
        if not self._check_paths(command):
            return "❌ 不允许访问工作目录外的路径"

        # 4. 执行命令
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.current_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "PWD": str(self.current_dir)},
            )

            output = result.stdout
            if result.stderr and cmd_name != "git":
                output += f"\n[stderr]\n{result.stderr}"

            if len(output) > self.max_output_size:
                output = output[:self.max_output_size]
                output += f"\n\n⚠️ 输出被截断 (超过 {self.max_output_size} 字节)"

            if result.returncode != 0:
                prefix = f"⚠️ 返回码: {result.returncode}\n\n"
                output = prefix + output

            return output if output.strip() else "✅ 命令执行成功 (无输出)"

        except subprocess.TimeoutExpired:
            return f"❌ 命令执行超时 (超过 {self.timeout} 秒)"
        except Exception as e:
            return f"❌ 命令执行失败: {e}"

    def _handle_cd(self, parts: List[str]) -> str:
        """处理 cd 命令"""
        if not self.allow_cd:
            return "❌ cd 命令已禁用"

        if len(parts) < 2:
            return f"当前目录: {self.current_dir}"

        target = parts[1]

        if target == "..":
            new_dir = self.current_dir.parent
        elif target == ".":
            new_dir = self.current_dir
        elif target == "~":
            new_dir = self.workspace
        elif target.startswith("/"):
            new_dir = Path(target).resolve()
        else:
            new_dir = (self.current_dir / target).resolve()

        # 沙箱检查
        try:
            new_dir.relative_to(self.workspace)
        except ValueError:
            return f"❌ 不允许访问工作目录外的路径: {new_dir}"

        if not new_dir.exists():
            return f"❌ 目录不存在: {new_dir}"
        if not new_dir.is_dir():
            return f"❌ 不是目录: {new_dir}"

        self.current_dir = new_dir
        return f"✅ 切换到: {self.current_dir}"

    def _check_paths(self, command: str) -> bool:
        """检查命令中的路径是否都在工作目录内"""
        import re
        # 查找可能的路径参数
        path_patterns = re.findall(r'(?:^|\s)([./~][^\s]*)', command)
        # 也检查绝对路径
        path_patterns += re.findall(r'(?:^|\s)(/[^\s]*)', command)

        for path_str in path_patterns:
            path_str = path_str.strip().strip('"').strip("'")
            if not path_str:
                continue
            try:
                p = (self.current_dir / path_str).resolve()
                p.relative_to(self.workspace)
            except (ValueError, OSError):
                # 对于 grep/find 的第一个参数（搜索模式），不是路径
                cmd_name = command.split()[0] if command.split() else ""
                if cmd_name in ("grep", "egrep", "fgrep"):
                    # grep 的第一个非选项参数是 pattern，可能是路径也可能不是
                    continue
                print(f"[TerminalTool] 阻止路径访问: {path_str}")
                return False
        return True

    # ========== 辅助 ==========

    def reset_dir(self):
        """重置当前目录到工作目录"""
        self.current_dir = self.workspace

    def pwd(self) -> str:
        """获取当前工作目录"""
        return str(self.current_dir)

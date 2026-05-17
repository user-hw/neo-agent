"""异步工具执行器"""
import asyncio
import concurrent.futures
from typing import Dict, List
from .registry import ToolRegistry


class AsyncToolExecutor:
    """异步工具执行器 - 支持并行执行多个工具"""

    def __init__(self, registry: ToolRegistry, max_workers: int = 4):
        self.registry = registry
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def execute_tool_async(self, tool_name: str, input_data: str) -> str:
        """异步执行单个工具"""
        loop = asyncio.get_event_loop()

        def _execute():
            return self.registry.execute_tool(tool_name, input_data)

        result = await loop.run_in_executor(self.executor, _execute)
        return result

    async def execute_tools_parallel(self, tasks: List[Dict[str, str]]) -> List[str]:
        """并行执行多个工具

        Args:
            tasks: 任务列表，每项包含 tool_name 和 input_data

        Returns:
            各工具的返回结果列表
        """
        print(f"🚀 开始并行执行 {len(tasks)} 个工具任务")

        async_tasks = [
            self.execute_tool_async(task["tool_name"], task["input_data"])
            for task in tasks
        ]

        results = await asyncio.gather(*async_tasks)
        print(f"✅ 所有工具任务执行完成")
        return list(results)

    def shutdown(self):
        """关闭线程池"""
        self.executor.shutdown(wait=True)

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

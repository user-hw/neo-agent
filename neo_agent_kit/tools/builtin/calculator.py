"""数学计算工具"""
import ast
import operator
import math
from typing import Dict, Any, List
from ..base import Tool, ToolParameter


class CalculatorTool(Tool):
    """安全的数学表达式计算工具

    支持:
    - 基本运算: +, -, *, /, //, %, **
    - 数学函数: sqrt, sin, cos, tan, log, log10, abs, pow
    - 常数: pi, e
    """

    # 支持的运算符
    _OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    # 支持的数学函数
    _FUNCTIONS = {
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'log10': math.log10,
        'abs': abs,
        'pow': pow,
        'round': round,
        'ceil': math.ceil,
        'floor': math.floor,
    }

    # 支持的常数
    _CONSTANTS = {
        'pi': math.pi,
        'e': math.e,
        'tau': math.tau,
        'inf': math.inf,
    }

    def __init__(self):
        super().__init__(
            name="calculator",
            description="安全的数学表达式计算工具。支持基本运算(+,-,*,/,**,%)和数学函数(sqrt,sin,cos,log,abs,round等)。"
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="要计算的数学表达式，如 '2 + 3 * 4' 或 'sqrt(16) + 2'",
                required=True
            )
        ]

    def run(self, parameters: Dict[str, Any]) -> str:
        expression = parameters.get("expression", "")
        if not expression or not expression.strip():
            return "❌ 错误: 表达式不能为空"

        try:
            # 解析 AST
            tree = ast.parse(expression.strip(), mode='eval')
            result = self._eval_node(tree.body)
            return str(result)
        except SyntaxError as e:
            return f"❌ 语法错误: {e}"
        except ZeroDivisionError:
            return "❌ 错误: 除数不能为零"
        except Exception as e:
            return f"❌ 计算失败: {e}"

    def _eval_node(self, node):
        """递归求值 AST 节点"""
        if isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self._OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
            return op(left, right)

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self._OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
            return op(operand)

        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name in self._FUNCTIONS:
                args = [self._eval_node(arg) for arg in node.args]
                return self._FUNCTIONS[func_name](*args)
            raise ValueError(f"不支持的函数: {func_name}")

        elif isinstance(node, ast.Name):
            if node.id in self._CONSTANTS:
                return self._CONSTANTS[node.id]
            raise ValueError(f"未知变量或常数: {node.id}")

        raise ValueError(f"不支持的表达式类型: {type(node).__name__}")

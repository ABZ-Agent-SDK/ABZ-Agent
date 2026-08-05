from pydantic import BaseModel
from abzagent.core.tools import Tool
import ast
import operator as op

class EvalSchema(BaseModel):
    expression: str

_ALLOWED_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg, ast.UAdd: lambda x: x,
}

def _eval_ast(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers allowed")
    if isinstance(node, ast.Num):  # py<=3.8
        return node.n
    if isinstance(node, ast.BinOp):
        return _ALLOWED_OPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp):
        return _ALLOWED_OPS[type(node.op)](_eval_ast(node.operand))
    raise ValueError("Invalid expression")

def safe_eval(expr: str) -> float:
    return _eval_ast(ast.parse(expr, mode="eval").body)

class MathTool(Tool):
    name = "calculator"  # 👈 match model’s call
    description = "Safely evaluate a basic arithmetic expression like '2+2*10'."
    schema = EvalSchema

    def run(self, **kwargs) -> str:
        return str(safe_eval(kwargs["expression"]))

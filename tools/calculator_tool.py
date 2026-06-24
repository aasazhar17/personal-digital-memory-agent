import ast
import operator
import asyncio
from typing import Dict, Any

class CalculatorTool:
    """
    A safe, production-ready calculator tool that evaluates basic arithmetic expressions.
    Uses AST (Abstract Syntax Tree) parsing rather than unsafe eval() to prevent security issues.
    """
    # Define supported operations
    _OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: lambda x: x
    }

    def __init__(self):
        pass

    async def calculate(self, expression: str) -> Dict[str, Any]:
        """
        Asynchronously evaluates a mathematical expression string safely.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._calculate_sync, expression)

    def _calculate_sync(self, expression: str) -> Dict[str, Any]:
        cleaned_expr = expression.replace("₹", "").replace("Rs.", "").replace("INR", "").replace(",", "").strip()
        try:
            # Parse the expression into an AST
            tree = ast.parse(cleaned_expr, mode='eval')
            result = self._evaluate_node(tree.body)
            # Format floating points nicely if they are decimals
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return {"success": True, "result": result, "expression": cleaned_expr}
        except Exception as e:
            return {"success": False, "error": f"Invalid expression: {str(e)}", "expression": cleaned_expr}

    def _evaluate_node(self, node) -> Any:
        """
        Recursively walks and evaluates mathematical nodes in the AST.
        """
        if isinstance(node, ast.Num): # Python < 3.8 compatibility
            return node.n
        elif isinstance(node, ast.Constant): # Python >= 3.8
            return node.value
        elif isinstance(node, ast.BinOp):
            left_val = self._evaluate_node(node.left)
            right_val = self._evaluate_node(node.right)
            op_type = type(node.op)
            if op_type in self._OPERATORS:
                return self._OPERATORS[op_type](left_val, right_val)
            raise TypeError(f"Unsupported binary operator: {op_type.__name__}")
        elif isinstance(node, ast.UnaryOp):
            operand_val = self._evaluate_node(node.operand)
            op_type = type(node.op)
            if op_type in self._OPERATORS:
                return self._OPERATORS[op_type](operand_val)
            raise TypeError(f"Unsupported unary operator: {op_type.__name__}")
        else:
            raise TypeError(f"Unsupported syntax expression: {type(node).__name__}")

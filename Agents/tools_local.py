"""Local tools — plain Python functions the agent can call.

The @tool decorator turns a function into something the model can invoke. Note
what Strands reads to build the tool schema the model sees:

  - the function NAME        -> the tool name
  - the TYPE HINTS           -> the parameter types
  - the DOCSTRING            -> the description + per-argument descriptions

So the docstring is not a comment here. It is prompt engineering. If the model
picks the wrong tool, the fix is almost always a better docstring.
"""

import ast
import operator
import random

from strands import tool

# Allowed operators for the calculator. We deliberately do NOT use eval(),
# which would let the model run arbitrary Python.
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(node):
    """Walk the parsed expression tree, allowing arithmetic and nothing else."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("Only numbers and + - * / // % ** are allowed.")


@tool
def calculate(expression: str) -> str:
    """Evaluate a arithmetic expression and return the result.

    Use this for any math the user asks for, instead of computing it yourself.

    Args:
        expression: An arithmetic expression, e.g. "(1234 * 17) / 3" or "2 ** 10".

    Returns:
        The numeric result, or an error message if the expression is invalid.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree.body)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as exc:
        return f"Error: could not evaluate {expression!r}. {exc}"


_JOKES = [
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "There are 10 kinds of people: those who understand binary and those who don't.",
    "I would tell you a UDP joke, but you might not get it.",
    "A SQL query walks into a bar, goes up to two tables and asks: may I join you?",
    "Why did the developer go broke? He used up all his cache.",
    "To understand recursion, you must first understand recursion.",
    "There are two hard problems in computer science: cache invalidation, "
    "naming things, and off-by-one errors.",
]


@tool
def tell_joke(topic: str = "programming") -> str:
    """Return a short joke.

    Args:
        topic: The requested topic. Only "programming" jokes are available;
            any other topic still returns a programming joke.

    Returns:
        A single joke as text.
    """
    return random.choice(_JOKES)


@tool
def roll_dice(sides: int = 6, count: int = 1) -> str:
    """Roll one or more dice and return the individual rolls and their total.

    Args:
        sides: Number of sides on each die. Must be at least 2.
        count: How many dice to roll. Must be between 1 and 100.

    Returns:
        The individual rolls and the total.
    """
    if sides < 2:
        return "Error: a die needs at least 2 sides."
    if not 1 <= count <= 100:
        return "Error: count must be between 1 and 100."

    rolls = [random.randint(1, sides) for _ in range(count)]
    return f"Rolled {count}d{sides}: {rolls} (total: {sum(rolls)})"

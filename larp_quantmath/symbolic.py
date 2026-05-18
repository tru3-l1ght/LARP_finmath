from __future__ import annotations

import sympy as sp


def _symbol(var: str) -> sp.Symbol:
    if not isinstance(var, str) or not var.strip():
        raise ValueError("var must be a non-empty string")

    return sp.Symbol(var)


def _expression(expr: str) -> sp.Expr:
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("expr must be a non-empty string")

    try:
        return sp.sympify(expr)
    except Exception as exc:
        raise ValueError(f"invalid symbolic expression: {expr}") from exc


def simplify_expr(expr: str) -> str:
    expression = _expression(expr)
    return str(sp.simplify(expression))


def derivative(expr: str, var: str = "x", order: int = 1) -> str:
    if order <= 0:
        raise ValueError("order must be positive")

    expression = _expression(expr)
    symbol = _symbol(var)

    return str(sp.diff(expression, symbol, order))


def integral(expr: str, var: str = "x") -> str:
    expression = _expression(expr)
    symbol = _symbol(var)

    return str(sp.integrate(expression, symbol))


def definite_integral(expr: str, var: str = "x", lower=0, upper=1) -> str:
    expression = _expression(expr)
    symbol = _symbol(var)

    lower_value = sp.sympify(lower)
    upper_value = sp.sympify(upper)

    return str(sp.integrate(expression, (symbol, lower_value, upper_value)))


def limit(expr: str, var: str = "x", point=0, direction: str = "+") -> str:
    expression = _expression(expr)
    symbol = _symbol(var)
    point_value = sp.sympify(point)

    if direction not in ["+", "-", "+-"]:
        raise ValueError("direction must be '+', '-', or '+-'")

    return str(sp.limit(expression, symbol, point_value, dir=direction))


def solve_expr(expr: str, var: str = "x") -> str:
    expression = _expression(expr)
    symbol = _symbol(var)

    return str(sp.solve(expression, symbol))


def taylor_series(expr: str, var: str = "x", point=0, order: int = 6) -> str:
    if order <= 0:
        raise ValueError("order must be positive")

    expression = _expression(expr)
    symbol = _symbol(var)
    point_value = sp.sympify(point)

    return str(sp.series(expression, symbol, point_value, order))

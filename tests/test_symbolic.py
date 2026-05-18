import larp_quantmath as lq


def test_simplify_expr():
    result = lq.simplify_expr("(x**2 - 1) / (x - 1)")
    assert result == "x + 1"


def test_derivative():
    result = lq.derivative("x**3", "x")
    assert result == "3*x**2"


def test_second_derivative():
    result = lq.derivative("x**3", "x", 2)
    assert result == "6*x"


def test_integral():
    result = lq.integral("2*x", "x")
    assert result == "x**2"


def test_definite_integral():
    result = lq.definite_integral("x", "x", 0, 1)
    assert result == "1/2"


def test_limit():
    result = lq.limit("sin(x)/x", "x", 0)
    assert result == "1"


def test_solve_expr():
    result = lq.solve_expr("x**2 - 4", "x")
    assert result == "[-2, 2]"

import math
import larp_quantmath as lq


def test_black_scholes_call():
    result = lq.black_scholes_call(100, 100, 0.05, 0.2, 1.0)
    assert math.isclose(result, 10.4506, rel_tol=1e-3)


def test_black_scholes_put():
    result = lq.black_scholes_put(100, 100, 0.05, 0.2, 1.0)
    assert math.isclose(result, 5.5735, rel_tol=1e-3)


def test_black_scholes_d1():
    result = lq.black_scholes_d1(100, 100, 0.05, 0.2, 1.0)
    assert math.isclose(result, 0.35, rel_tol=1e-9)


def test_black_scholes_d2():
    result = lq.black_scholes_d2(100, 100, 0.05, 0.2, 1.0)
    assert math.isclose(result, 0.15, rel_tol=1e-9)

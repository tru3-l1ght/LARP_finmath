import math
import larp_quantmath as lq


def test_monte_carlo_call_close_to_black_scholes():
    bs = lq.black_scholes_call(100, 100, 0.05, 0.2, 1.0)
    mc = lq.monte_carlo_call(100, 100, 0.05, 0.2, 1.0, 100_000, 42)

    assert math.isclose(mc, bs, rel_tol=0.03)


def test_monte_carlo_put_close_to_black_scholes():
    bs = lq.black_scholes_put(100, 100, 0.05, 0.2, 1.0)
    mc = lq.monte_carlo_put(100, 100, 0.05, 0.2, 1.0, 100_000, 42)

    assert math.isclose(mc, bs, rel_tol=0.03)


def test_monte_carlo_is_seeded():
    result_1 = lq.monte_carlo_call(100, 100, 0.05, 0.2, 1.0, 10_000, 123)
    result_2 = lq.monte_carlo_call(100, 100, 0.05, 0.2, 1.0, 10_000, 123)

    assert result_1 == result_2

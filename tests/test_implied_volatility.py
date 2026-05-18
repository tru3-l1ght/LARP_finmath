import math
import larp_quantmath as lq


def test_implied_volatility_call():
    market_price = lq.black_scholes_call(100, 100, 0.05, 0.2, 1.0)

    result = lq.implied_volatility_call(
        market_price,
        100,
        100,
        0.05,
        1.0,
        1e-8,
        100,
    )

    assert math.isclose(result, 0.2, rel_tol=1e-5)


def test_implied_volatility_put():
    market_price = lq.black_scholes_put(100, 100, 0.05, 0.2, 1.0)

    result = lq.implied_volatility_put(
        market_price,
        100,
        100,
        0.05,
        1.0,
        1e-8,
        100,
    )

    assert math.isclose(result, 0.2, rel_tol=1e-5)

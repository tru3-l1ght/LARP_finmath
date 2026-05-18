import math
import larp_quantmath as lq


def test_sma():
    result = lq.sma([1, 2, 3, 4, 5], 3)
    assert result == [2.0, 3.0, 4.0]


def test_ema_length():
    result = lq.ema([1, 2, 3, 4, 5], 3)
    assert len(result) == 3


def test_ema_first_value_is_sma():
    result = lq.ema([1, 2, 3, 4, 5], 3)
    assert math.isclose(result[0], 2.0)


def test_ema_values():
    result = lq.ema([1, 2, 3, 4, 5], 3)
    assert math.isclose(result[1], 3.0)
    assert math.isclose(result[2], 4.0)


def test_rsi_length():
    prices = [44, 44.15, 43.9, 44.35, 44.8, 45.0, 44.7, 45.2]
    result = lq.rsi(prices, 3)
    assert len(result) == len(prices) - 3


def test_rsi_bounds():
    prices = [44, 44.15, 43.9, 44.35, 44.8, 45.0, 44.7, 45.2]
    result = lq.rsi(prices, 3)

    for value in result:
        assert 0 <= value <= 100


def test_rolling_volatility():
    returns = [0.01, -0.02, 0.015, 0.005, -0.01]
    result = lq.rolling_volatility(returns, 3, 252)

    assert len(result) == 3
    assert all(value > 0 for value in result)

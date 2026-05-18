import math
import larp_quantmath as lq


def test_total_return():
    result = lq.total_return([100, 110, 121])
    assert math.isclose(result, 0.21)


def test_cagr():
    result = lq.cagr([100, 121], 2)
    assert math.isclose(result, 0.10)


def test_drawdown_series():
    result = lq.drawdown_series([100, 120, 90, 130, 100])
    expected = [0.0, 0.0, -0.25, 0.0, -30 / 130]

    for a, b in zip(result, expected):
        assert math.isclose(a, b)


def test_calmar_ratio():
    result = lq.calmar_ratio([100, 120, 90, 130], 1)
    expected = 0.30 / 0.25

    assert math.isclose(result, expected)


def test_win_rate():
    result = lq.win_rate([0.01, -0.02, 0.03, 0.00])
    assert math.isclose(result, 0.5)


def test_profit_factor():
    result = lq.profit_factor([0.10, -0.05, 0.20, -0.10])
    assert math.isclose(result, 2.0)


def test_average_return():
    result = lq.average_return([0.01, 0.02, 0.03])
    assert math.isclose(result, 0.02)


def test_annualized_return():
    result = lq.annualized_return([0.01] * 12, 12)
    expected = (1.01 ** 12) - 1

    assert math.isclose(result, expected)


def test_annualized_volatility():
    returns = [0.01, -0.02, 0.015, 0.005, -0.01]
    result = lq.annualized_volatility(returns, 252)

    expected = lq.std_dev(returns, True) * math.sqrt(252)

    assert math.isclose(result, expected)

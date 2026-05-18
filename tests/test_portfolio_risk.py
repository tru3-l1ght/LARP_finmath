import math
import larp_quantmath as lq


def test_portfolio_return():
    result = lq.portfolio_return([0.10, 0.20], [0.6, 0.4])
    assert math.isclose(result, 0.14)


def test_portfolio_volatility():
    cov = [
        [0.04, 0.006],
        [0.006, 0.09],
    ]
    weights = [0.5, 0.5]

    result = lq.portfolio_volatility(cov, weights)

    expected_variance = 0.25 * 0.04 + 2 * 0.25 * 0.006 + 0.25 * 0.09
    expected_volatility = math.sqrt(expected_variance)

    assert math.isclose(result, expected_volatility)


def test_sharpe_ratio():
    result = lq.sharpe_ratio(0.12, 0.02, 0.20)
    assert math.isclose(result, 0.5)


def test_historical_var():
    returns = [-0.10, -0.05, -0.02, 0.01, 0.03, 0.04]
    result = lq.historical_var(returns, 0.95)

    assert math.isclose(result, 0.10)


def test_max_drawdown():
    prices = [100, 120, 90, 95, 130, 100]
    result = lq.max_drawdown(prices)

    assert math.isclose(result, 0.25)

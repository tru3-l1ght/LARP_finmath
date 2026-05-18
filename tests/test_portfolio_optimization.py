import math
import larp_quantmath as lq


def test_normalize_weights():
    result = lq.normalize_weights([2, 3, 5])
    assert result == [0.2, 0.3, 0.5]


def test_equal_weight_portfolio():
    result = lq.equal_weight_portfolio(4)
    assert result == [0.25, 0.25, 0.25, 0.25]


def test_portfolio_variance():
    cov = [
        [0.04, 0.01],
        [0.01, 0.09],
    ]
    weights = [0.5, 0.5]

    result = lq.portfolio_variance(cov, weights)

    expected = 0.5 * 0.5 * 0.04 + 2 * 0.5 * 0.5 * 0.01 + 0.5 * 0.5 * 0.09

    assert math.isclose(result, expected)


def test_minimum_variance_two_asset_weights_sum_to_one():
    result = lq.minimum_variance_two_asset_weights(0.04, 0.09, 0.01)

    assert math.isclose(sum(result), 1.0)


def test_minimum_variance_two_asset_weights_values():
    result = lq.minimum_variance_two_asset_weights(0.04, 0.09, 0.01)

    expected_a = (0.09 - 0.01) / (0.04 + 0.09 - 2 * 0.01)
    expected_b = 1 - expected_a

    assert math.isclose(result[0], expected_a)
    assert math.isclose(result[1], expected_b)


def test_risk_parity_two_asset_weights_sum_to_one():
    result = lq.risk_parity_two_asset_weights(0.2, 0.4)

    assert math.isclose(sum(result), 1.0)


def test_risk_parity_two_asset_weights_values():
    result = lq.risk_parity_two_asset_weights(0.2, 0.4)

    assert math.isclose(result[0], 2 / 3)
    assert math.isclose(result[1], 1 / 3)


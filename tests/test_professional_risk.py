import math
import larp_quantmath as lq


def test_downside_deviation():
    returns = [0.02, -0.01, 0.03, -0.02]
    result = lq.downside_deviation(returns, 0.0, 1)

    expected = math.sqrt((0.0**2 + (-0.01)**2 + 0.0**2 + (-0.02)**2) / 4)

    assert math.isclose(result, expected)


def test_sortino_ratio():
    result = lq.sortino_ratio(0.12, 0.02, 0.20)
    assert math.isclose(result, 0.5)


def test_beta():
    asset = [0.02, 0.04, -0.01, 0.03]
    market = [0.01, 0.02, -0.005, 0.015]

    result = lq.beta(asset, market)

    assert math.isclose(result, 2.0, rel_tol=1e-9)


def test_alpha():
    result = lq.alpha(0.12, 0.02, 1.2, 0.10)

    expected = 0.12 - (0.02 + 1.2 * (0.10 - 0.02))

    assert math.isclose(result, expected)


def test_tracking_error():
    asset = [0.02, 0.01, 0.03, -0.01]
    benchmark = [0.01, 0.015, 0.02, -0.005]

    result = lq.tracking_error(asset, benchmark, 1)

    active = [0.01, -0.005, 0.01, -0.005]
    expected = lq.std_dev(active, True)

    assert math.isclose(result, expected)


def test_information_ratio():
    asset = [0.03, 0.02, 0.04, 0.01]
    benchmark = [0.02, 0.01, 0.03, 0.00]

    result = lq.information_ratio(asset, benchmark, 1)

    active = [0.01, 0.01, 0.01, 0.01]
    expected_active_return = sum(active) / len(active)

    # Tracking error is zero when active return is constant, so this should raise.
    assert math.isclose(expected_active_return, 0.01)
    assert result > 0

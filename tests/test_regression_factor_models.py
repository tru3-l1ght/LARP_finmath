import math
import larp_quantmath as lq


def test_linear_regression():
    slope, intercept = lq.linear_regression([1, 2, 3], [2, 4, 6])

    assert math.isclose(slope, 2.0)
    assert math.isclose(intercept, 0.0)


def test_predict_linear():
    result = lq.predict_linear([1, 2, 3], 2, 1)

    assert result == [3.0, 5.0, 7.0]


def test_residuals():
    result = lq.residuals([1, 2, 3], [2, 4, 6], 2, 0)

    assert result == [0.0, 0.0, 0.0]


def test_r_squared_perfect_fit():
    result = lq.r_squared([1, 2, 3], [2, 4, 6], 2, 0)

    assert math.isclose(result, 1.0)


def test_r_squared_imperfect_fit():
    slope, intercept = lq.linear_regression([1, 2, 3, 4], [2, 4, 5, 8])
    result = lq.r_squared([1, 2, 3, 4], [2, 4, 5, 8], slope, intercept)

    assert 0.0 <= result <= 1.0


def test_capm_expected_return():
    result = lq.capm_expected_return(0.02, 1.2, 0.10)

    expected = 0.02 + 1.2 * (0.10 - 0.02)

    assert math.isclose(result, expected)


def test_factor_exposure():
    asset = [0.02, 0.04, -0.01, 0.03]
    factor = [0.01, 0.02, -0.005, 0.015]

    result = lq.factor_exposure(asset, factor)

    assert math.isclose(result, 2.0, rel_tol=1e-9)

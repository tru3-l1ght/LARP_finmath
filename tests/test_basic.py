import math
import larp_quantmath as lq


def test_mean():
    assert lq.mean([1, 2, 3, 4, 5]) == 3.0


def test_variance_sample():
    assert math.isclose(lq.variance([1, 2, 3, 4, 5], True), 2.5)


def test_std_dev_sample():
    assert math.isclose(lq.std_dev([1, 2, 3, 4, 5], True), math.sqrt(2.5))


def test_simple_returns():
    result = lq.simple_returns([100, 110, 121])
    assert result == [0.1, 0.1]


def test_log_returns():
    result = lq.log_returns([100, 110])
    assert math.isclose(result[0], math.log(1.1))


def test_correlation():
    result = lq.correlation([1, 2, 3], [2, 4, 6])
    assert math.isclose(result, 1.0)
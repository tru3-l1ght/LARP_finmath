import math
import larp_quantmath as lq


def test_macd_shapes():
    prices = [float(i) for i in range(1, 60)]

    macd_line, signal_line, histogram = lq.macd(prices, 12, 26, 9)

    assert len(macd_line) == len(prices) - 26 + 1
    assert len(signal_line) == len(macd_line) - 9 + 1
    assert len(histogram) == len(signal_line)


def test_macd_histogram_relationship():
    prices = [float(i) for i in range(1, 80)]

    macd_line, signal_line, histogram = lq.macd(prices, 12, 26, 9)

    aligned_macd = macd_line[-len(signal_line):]

    for m, s, h in zip(aligned_macd, signal_line, histogram):
        assert math.isclose(h, m - s)


def test_bollinger_bands_shapes():
    prices = [1, 2, 3, 4, 5, 6]

    middle, upper, lower = lq.bollinger_bands(prices, 3, 2.0)

    assert len(middle) == 4
    assert len(upper) == 4
    assert len(lower) == 4


def test_bollinger_bands_order():
    prices = [1, 2, 3, 4, 5, 6]

    middle, upper, lower = lq.bollinger_bands(prices, 3, 2.0)

    for m, u, l in zip(middle, upper, lower):
        assert u > m
        assert m > l


def test_z_score():
    result = lq.z_score([1, 2, 3, 4, 5], 5)

    expected = (5 - 3) / math.sqrt(2.5)

    assert math.isclose(result, expected)

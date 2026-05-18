import math
import larp_quantmath as lq


def test_yield_to_maturity_at_par():
    result = lq.yield_to_maturity(
        1000,
        1000,
        0.05,
        5,
        1,
        1e-9,
        100,
    )

    assert math.isclose(result, 0.05, rel_tol=1e-6)


def test_zero_coupon_bond_price():
    result = lq.zero_coupon_bond_price(1000, 0.05, 2)

    expected = 1000 / (1.05**2)

    assert math.isclose(result, expected)


def test_dv01_positive():
    result = lq.dv01(1000, 0.05, 0.05, 5, 1)

    assert result > 0


def test_forward_rate():
    result = lq.forward_rate(0.03, 0.05, 1, 2)

    expected = ((1.05**2) / 1.03) - 1

    assert math.isclose(result, expected)

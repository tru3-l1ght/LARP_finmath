import math
import larp_quantmath as lq


def test_bond_price_at_par():
    result = lq.bond_price(1000, 0.05, 0.05, 5, 1)
    assert math.isclose(result, 1000.0, rel_tol=1e-9)


def test_bond_price_discount():
    result = lq.bond_price(1000, 0.04, 0.06, 5, 1)
    assert result < 1000


def test_bond_price_premium():
    result = lq.bond_price(1000, 0.06, 0.04, 5, 1)
    assert result > 1000


def test_current_yield():
    result = lq.current_yield(1000, 0.05, 950)
    assert math.isclose(result, 50 / 950)


def test_macaulay_duration_positive():
    result = lq.macaulay_duration(1000, 0.05, 0.05, 5, 1)
    assert result > 0
    assert result < 5


def test_modified_duration_less_than_macaulay():
    mac = lq.macaulay_duration(1000, 0.05, 0.05, 5, 1)
    mod = lq.modified_duration(1000, 0.05, 0.05, 5, 1)

    assert mod < mac


def test_bond_convexity_positive():
    result = lq.bond_convexity(1000, 0.05, 0.05, 5, 1)
    assert result > 0

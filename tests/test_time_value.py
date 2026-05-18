import math
import larp_quantmath as lq


def test_present_value():
    result = lq.present_value(110, 0.10, 1)
    assert math.isclose(result, 100.0)


def test_future_value():
    result = lq.future_value(100, 0.10, 1)
    assert math.isclose(result, 110.0)


def test_net_present_value():
    result = lq.net_present_value(0.10, [-1000, 500, 700])
    expected = -1000 + 500 / 1.10 + 700 / (1.10**2)

    assert math.isclose(result, expected)


def test_internal_rate_of_return():
    result = lq.internal_rate_of_return([-100, 110], 1e-9, 100)
    assert math.isclose(result, 0.10, rel_tol=1e-6)


def test_loan_payment_zero_rate():
    result = lq.loan_payment(1200, 0.0, 1, 12)
    assert math.isclose(result, 100.0)


def test_loan_payment_positive_rate():
    result = lq.loan_payment(1000, 0.12, 1, 12)

    monthly_rate = 0.12 / 12
    expected = 1000 * monthly_rate / (1 - (1 + monthly_rate) ** -12)

    assert math.isclose(result, expected)


def test_annuity_payment_zero_rate():
    result = lq.annuity_payment(1200, 0.0, 12)
    assert math.isclose(result, 100.0)


def test_annuity_payment_positive_rate():
    result = lq.annuity_payment(1000, 0.01, 12)
    expected = 1000 * 0.01 / (1 - (1.01) ** -12)

    assert math.isclose(result, expected)

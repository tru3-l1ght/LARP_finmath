import math
import larp_quantmath as lq


def test_delta_call():
    result = lq.delta_call(100, 100, 0.05, 0.2, 1.0)
    assert math.isclose(result, 0.6368, rel_tol=1e-3)


def test_delta_put():
    result = lq.delta_put(100, 100, 0.05, 0.2, 1.0)
    assert math.isclose(result, -0.3632, rel_tol=1e-3)


def test_gamma():
    result = lq.gamma(100, 100, 0.05, 0.2, 1.0)
    assert math.isclose(result, 0.01876, rel_tol=1e-3)


def test_vega():
    result = lq.vega(100, 100, 0.05, 0.2, 1.0)
    assert math.isclose(result, 37.5240, rel_tol=1e-3)


def test_theta_call():
    result = lq.theta_call(100, 100, 0.05, 0.2, 1.0)
    assert math.isclose(result, -6.4140, rel_tol=1e-3)


def test_theta_put():
    result = lq.theta_put(100, 100, 0.05, 0.2, 1.0)
    assert math.isclose(result, -1.6579, rel_tol=1e-3)


def test_rho_call():
    result = lq.rho_call(100, 100, 0.05, 0.2, 1.0)
    assert math.isclose(result, 53.2325, rel_tol=1e-3)


def test_rho_put():
    result = lq.rho_put(100, 100, 0.05, 0.2, 1.0)
    assert math.isclose(result, -41.8905, rel_tol=1e-3)

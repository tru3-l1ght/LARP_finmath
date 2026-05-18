import math
import pytest
import larp_quantmath as lq


def test_matrix_add():
    result = lq.matrix_add([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    assert result == [[6.0, 8.0], [10.0, 12.0]]


def test_matrix_subtract():
    result = lq.matrix_subtract([[5, 6], [7, 8]], [[1, 2], [3, 4]])
    assert result == [[4.0, 4.0], [4.0, 4.0]]


def test_scalar_multiply_matrix():
    result = lq.scalar_multiply_matrix([[1, 2], [3, 4]], 2)
    assert result == [[2.0, 4.0], [6.0, 8.0]]


def test_vector_norm():
    result = lq.vector_norm([3, 4])
    assert math.isclose(result, 5.0)


def test_normalize_vector():
    result = lq.normalize_vector([3, 4])
    assert math.isclose(result[0], 0.6)
    assert math.isclose(result[1], 0.8)


def test_normalize_zero_vector_raises():
    with pytest.raises(ValueError):
        lq.normalize_vector([0, 0])


def test_determinant_2x2():
    result = lq.determinant_2x2([[1, 2], [3, 4]])
    assert math.isclose(result, -2.0)


def test_inverse_2x2():
    result = lq.inverse_2x2([[4, 7], [2, 6]])

    expected = [[0.6, -0.7], [-0.2, 0.4]]

    for row_result, row_expected in zip(result, expected):
        for a, b in zip(row_result, row_expected):
            assert math.isclose(a, b)

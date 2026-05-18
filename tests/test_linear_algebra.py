import math
import larp_quantmath as lq


def test_dot_product():
    result = lq.dot_product([1, 2, 3], [4, 5, 6])
    assert math.isclose(result, 32.0)


def test_transpose():
    result = lq.transpose([[1, 2, 3], [4, 5, 6]])
    assert result == [[1, 4], [2, 5], [3, 6]]


def test_matrix_multiply():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]

    result = lq.matrix_multiply(a, b)

    assert result == [[19.0, 22.0], [43.0, 50.0]]


def test_matrix_vector_multiply():
    matrix = [[1, 2], [3, 4]]
    vector = [5, 6]

    result = lq.matrix_vector_multiply(matrix, vector)

    assert result == [17.0, 39.0]


def test_identity_matrix():
    result = lq.identity_matrix(3)

    assert result == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def test_covariance_matrix_shape():
    data = [
        [1, 2, 3, 4],
        [2, 4, 6, 8],
        [4, 3, 2, 1],
    ]

    result = lq.covariance_matrix(data, True)

    assert len(result) == 3
    assert len(result[0]) == 3


def test_covariance_matrix_values():
    data = [
        [1, 2, 3],
        [2, 4, 6],
    ]

    result = lq.covariance_matrix(data, True)

    assert math.isclose(result[0][0], lq.variance([1, 2, 3], True))
    assert math.isclose(result[1][1], lq.variance([2, 4, 6], True))
    assert math.isclose(result[0][1], result[1][0])


def test_correlation_matrix_diagonal():
    data = [
        [1, 2, 3, 4],
        [2, 4, 6, 8],
    ]

    result = lq.correlation_matrix(data)

    assert math.isclose(result[0][0], 1.0)
    assert math.isclose(result[1][1], 1.0)


def test_correlation_matrix_values():
    data = [
        [1, 2, 3, 4],
        [2, 4, 6, 8],
    ]

    result = lq.correlation_matrix(data)

    assert math.isclose(result[0][1], 1.0)
    assert math.isclose(result[1][0], 1.0)

"""Unit tests for small machine-learning utility functions.

These tests compare the project's metric implementations against
scikit-learn references and test other ML utilities.
"""

import numpy as np
import pytest
from sklearn import metrics

from discrete1.utils import machine_learning as ml

np.random.seed(42)


@pytest.mark.math
def test_combine_flux_reaction():
    # Test data setup
    flux = np.array([[[1, 2], [3, 4]]])  # 1 iteration, 2 cells, 2 groups
    xs_matrix = np.array([[[0.5, 0.3], [0.2, 0.4]]])  # 1 material, 2x2 matrix
    medium_map = np.array([0, 0])  # Both cells use material 0
    labels = np.array([1.0, 2.0])  # Labels for each cell

    result = ml._combine_flux_reaction(flux, xs_matrix, medium_map, labels)

    # Expected shape: (2, iterations*cells, groups+1)
    assert result.shape == (2, 2, 3)

    # Check labels are in first column
    assert np.array_equal(result[0, :, 0], labels)
    assert np.array_equal(result[1, :, 0], labels)

    # Check flux values
    assert np.array_equal(result[0, 0, 1:], flux[0, 0])
    assert np.array_equal(result[0, 1, 1:], flux[0, 1])

    # Check reaction rates (matrix multiplication of flux and xs_matrix)
    expected_rates_0 = flux[0, 0] @ xs_matrix[0].T
    expected_rates_1 = flux[0, 1] @ xs_matrix[0].T
    assert np.allclose(result[1, 0, 1:], expected_rates_0)
    assert np.allclose(result[1, 1, 1:], expected_rates_1)


@pytest.mark.math
def test_min_max_normalization():
    # Test basic normalization
    data = np.array([[1, 2, 3], [4, 5, 6]])
    normalized = ml.min_max_normalization(data)
    expected = np.array([[0, 0.5, 1], [0, 0.5, 1]])
    assert np.allclose(normalized, expected)

    # Test with NaN and inf values
    data_with_nan = np.array([[1, np.nan, 3], [4, np.inf, 6]])
    normalized, high, low = ml.min_max_normalization(data_with_nan, verbose=True)
    assert not np.any(np.isnan(normalized))
    assert not np.any(np.isinf(normalized))
    assert np.all((normalized >= 0) & (normalized <= 1))


@pytest.mark.math
def test_root_normalization():
    data = np.array([[1, 4, 9], [16, 25, 36]])
    # Test square root
    sqrt_norm = ml.root_normalization(data, 0.5)
    expected_sqrt = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(sqrt_norm, expected_sqrt)

    # Test cube root
    cbrt_norm = ml.root_normalization(data, 1 / 3)
    expected_cbrt = np.array([[1, 1.5874, 2.0801], [2.5198, 2.924, 3.3019]])
    assert np.allclose(cbrt_norm, expected_cbrt, rtol=1e-4)


@pytest.mark.math
def test_mean_absolute_error():
    # Create 2 different arrays
    n = 100
    y_true = np.sort(2 * np.random.rand(n, n))
    y_pred = y_true + 0.1 * np.random.rand(n, n)
    # Get sklearn function
    reference = []
    for row in range(n):
        reference.append(metrics.mean_absolute_error(y_true[row], y_pred[row]))
    reference = np.array(reference)
    # Personal implementation
    approx = ml.mean_absolute_error(y_true, y_pred)
    assert np.array_equal(reference, approx), "Implementation doesn't match SKLearn"


@pytest.mark.math
def test_mean_squared_error():
    # Create 2 different arrays
    n = 100
    y_true = np.sort(2 * np.random.rand(n, n))
    y_pred = y_true + 0.1 * np.random.rand(n, n)
    # Get sklearn function
    reference = []
    for row in range(n):
        reference.append(metrics.mean_squared_error(y_true[row], y_pred[row]))
    reference = np.array(reference)
    # Personal implementation
    approx = ml.mean_squared_error(y_true, y_pred)
    assert np.array_equal(reference, approx), "Implementation doesn't match SKLearn"


@pytest.mark.math
def test_root_mean_squared_error():
    y_true = np.array([[1, 2, 3], [4, 5, 6]])
    y_pred = np.array([[1.1, 2.2, 2.8], [4.2, 4.8, 6.2]])
    rmse = ml.root_mean_squared_error(y_true, y_pred)

    # Calculate expected values manually
    mse1 = np.mean([(1.1 - 1) ** 2, (2.2 - 2) ** 2, (2.8 - 3) ** 2])
    mse2 = np.mean([(4.2 - 4) ** 2, (4.8 - 5) ** 2, (6.2 - 6) ** 2])
    expected = np.array([np.sqrt(mse1), np.sqrt(mse2)])

    assert np.allclose(rmse, expected)


@pytest.mark.math
def test_explained_variance_score():
    # Create 2 different arrays
    n = 100
    y_true = np.sort(2 * np.random.rand(n, n))
    y_pred = y_true + 0.1 * np.random.rand(n, n)
    # Get sklearn function
    reference = []
    for row in range(n):
        reference.append(metrics.explained_variance_score(y_true[row], y_pred[row]))
    reference = np.array(reference)
    # Personal implementation
    approx = ml.explained_variance_score(y_true, y_pred)
    assert np.array_equal(reference, approx), "Implementation doesn't match SKLearn"


@pytest.mark.math
def test_r2_score():
    # Create 2 different arrays
    n = 100
    y_true = np.sort(2 * np.random.rand(n, n))
    y_pred = y_true + 0.1 * np.random.rand(n, n)
    # Get sklearn function
    reference = []
    for row in range(n):
        reference.append(metrics.r2_score(y_true[row], y_pred[row]))
    reference = np.array(reference)
    # Personal implementation
    approx = ml.r2_score(y_true, y_pred)
    assert np.array_equal(reference, approx), "Implementation doesn't match SKLearn"


@pytest.mark.math
def test_update_cross_sections():
    # Create test data
    xs_matrix = np.array(
        [
            [[1, 2], [3, 4]],  # Material 0
            [[5, 6], [7, 8]],  # Material 1
            [[9, 10], [11, 12]],  # Material 2
        ]
    )
    model_idx = [0, 2]  # Materials 0 and 2 should be updated

    result = ml.update_cross_sections(xs_matrix, model_idx)

    # Check shape is preserved
    assert result.shape == xs_matrix.shape

    # Check materials in model_idx have summed first axis
    assert np.array_equal(result[0, :, 0], np.sum(xs_matrix[0], axis=0))
    assert np.array_equal(result[2, :, 0], np.sum(xs_matrix[2], axis=0))

    # Check material not in model_idx is unchanged
    assert np.array_equal(result[1], xs_matrix[1])

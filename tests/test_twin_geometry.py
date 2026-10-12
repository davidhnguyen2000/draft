"""Hull geometry must be unchanged and work with NumPy 2's warnings enabled."""
import warnings

import numpy as np
import pytest

from draft.twins.measure import _hull_2d


@pytest.mark.parametrize("points,expected", [
    ([[1, 1], [0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [1, 1]],
     [[0, 0], [1, 0], [1, 1], [0, 1]]),
    ([[2, 0], [1, 0], [0, 0], [1, 0]], [[0, 0], [2, 0]]),
    ([[0, 0], [1, 1]], [[0, 0], [1, 1]]),
])
def test_hull_preserves_boundary_without_deprecated_cross(points, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        hull = _hull_2d(np.asarray(points, dtype=float))
    np.testing.assert_array_equal(hull, expected)

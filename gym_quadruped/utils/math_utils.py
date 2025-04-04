from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def skew(x):
    """Skew symmetric matrix from a 3D vector."""
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def homogenous_transform(vec: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transformation matrix to a 3D vector.

    Args:
    ----
        vec: (3,) vector
        X: (4, 4) homogeneous transformation matrix

    Returns:
    -------
        (3,) vector
    """
    assert vec.flatten().shape == (3,), f'Expected 3D vector, got {vec} of shape {vec.shape}'
    assert X.shape == (4, 4) and X[3, 3] == 1, f'Expected homogeneous transformation matrix, got {X}'
    pos_hom = np.concatenate([vec.flatten(), [1]])
    X_pos_hom = X @ pos_hom
    return X_pos_hom[:3]


def hom2pos_quatwxyz(X: np.ndarray):
    """Convert a homogeneous transformation matrix in SE(3) to position and quaternion representation."""
    assert X.shape == (4, 4), f'Expected homogeneous transformation matrix, got {X}'
    return Rotation.from_matrix(X[:3, :3]).as_quat(scalar_first=True)


def angle_between_vectors(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """Calculate the angle between two vectors.

    Args:
        vector1 (tuple): vector 1
        vector2 (tuple): vector 2
    Returns:
        Angle in radians
    """
    """dot_product = np.dot(vector1, vector2) magnitude_vector1 = np.linalg.norm(vector1)

    magnitude_vector2 = np.linalg.norm(vector2) return np.arccos(dot_product / (magnitude_vector1 * magnitude_vector2))
    """
    vector_diff = np.asarray(vector2) - vector1
    return np.arctan2(vector_diff[1], vector_diff[0])


def _process_range(values):
    # IF vallue is a number
    if isinstance(values, (int, float, np.number)):
        return (values, values)  # Min, Max
    elif isinstance(values, (tuple, list, np.ndarray)):
        assert len(values) == 2, f'Invalid range values, expected (min, max) got: {values}'
        return values

import numpy as np
from scipy.spatial.transform import Rotation


def skew(x):
    """Skew symmetric matrix from a 3D vector."""
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


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
    assert vec.flatten().shape == (3,), f"Expected 3D vector, got {vec} of shape {vec.shape}"
    assert X.shape == (4, 4) and X[3, 3] == 1, f"Expected homogeneous transformation matrix, got {X}"
    pos_hom = np.concatenate([vec.flatten(), [1]])
    X_pos_hom = X @ pos_hom
    return X_pos_hom[:3]


def hom2pos_quatwxyz(X: np.ndarray):
    assert X.shape == (4, 4), f"Expected homogeneous transformation matrix, got {X}"
    pos = X[:3, 3]
    quat_xyzw = Rotation.from_matrix(X[:3, :3]).as_quat()

# def vector_to_quaternion(vector):
#     # Normalize the vector (if not already normalized)
#     norm = np.linalg.norm(vector)
#     if norm == 0:
#         raise ValueError("Zero vector cannot be converted to a quaternion")
    
#     unit_vector = vector / norm

#     # Define the reference axis (e.g., the x-axis)
#     ref_axis = np.array([0.0, 0.0, 1.0])

#     # Calculate the angle between the unit_vector and the ref_axis
#     dot_product = np.dot(ref_axis, unit_vector)
#     angle = np.arccos(dot_product)

#     # Calculate the axis of rotation (cross product)
#     axis_of_rotation = np.cross(ref_axis, unit_vector)
#     if np.linalg.norm(axis_of_rotation) == 0:  # If the vector is aligned with ref_axis
#         axis_of_rotation = np.array([0.0, 0.0, 1.0])  # Choose an arbitrary axis

#     # Normalize the axis of rotation
#     axis_of_rotation /= np.linalg.norm(axis_of_rotation)

#     # Convert the axis-angle representation to a quaternion
#     quaternion = Rotation.from_rotvec(angle * axis_of_rotation).as_quat()

#     return quaternion

def vector_to_quaternion(vector):
    # Normalize the vector (if not already normalized)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Zero vector cannot be converted to a quaternion")
    
    unit_vector = vector / norm

    # Define the reference axis (e.g., the z-axis, which is a common "forward" direction in 3D space)
    ref_axis = np.array([0.0, 0.0, 1.0])

    # Calculate the axis of rotation (cross product)
    axis_of_rotation = np.cross(ref_axis, unit_vector)
    axis_norm = np.linalg.norm(axis_of_rotation)
    
    if axis_norm < 1e-6:  # If the vector is aligned with ref_axis
        if np.allclose(unit_vector, ref_axis):
            # No rotation needed, return identity quaternion
            return np.array([0.0, 0.0, 0.0, 1.0])
        else:
            # 180 degrees rotation about any axis orthogonal to ref_axis
            axis_of_rotation = np.array([0.0, 0.0, 1.0])  # Arbitrarily chosen orthogonal axis
            angle = np.pi
    else:
        # Calculate the angle between the unit_vector and the ref_axis
        angle = np.arccos(np.clip(np.dot(ref_axis, unit_vector), -1.0, 1.0))

    # Normalize the axis of rotation
    axis_of_rotation /= axis_norm

    # Convert the axis-angle representation to a quaternion
    quaternion = Rotation.from_rotvec(angle * axis_of_rotation).as_quat()

    return quaternion #[x,y,z,w]
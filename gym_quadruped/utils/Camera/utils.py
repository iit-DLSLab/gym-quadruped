from typing import List, Union

import numpy as np
import spatialmath as sm #pip install spatialmath-python
import spatialmath.base as smb
import pandas as pd


def make_tf(
    pos: Union[np.ndarray, list] = [0, 0, 0],
    ori: Union[np.ndarray, sm.SE3, sm.SO3] = [1, 0, 0, 0],
) -> sm.SE3:
    """
    Create a SE3 transformation matrix.

    Parameters:
    - pos (np.ndarray): Translation vector [x, y, z].
    - ori (Union[np.ndarray, SE3]): Orientation, can be a rotation matrix, quaternion, or SE3 object.

    Returns:
    - SE3: Transformation matrix.
    """

    if isinstance(ori, list):
        ori = np.array(ori)

    if isinstance(ori, sm.SO3):
        ori = ori.R

    if isinstance(pos, sm.SE3):
        pose = pos
        pos = pose.t
        ori = pose.R

    if len(ori) == 9:
        ori = np.reshape(ori, (3, 3))

    # Convert ori to SE3 if it's already a rotation matrix or a quaternion
    if isinstance(ori, np.ndarray):
        if ori.shape == (3, 3):  # Assuming ori is a rotation matrix
            ori = ori
        elif ori.shape == (4,):  # Assuming ori is a quaternion
            ori = sm.UnitQuaternion(s=ori[0], v=ori[1:]).R
        elif ori.shape == (3,):  # Assuming ori is rpy
            ori = sm.SE3.Eul(ori, unit="rad").R

    T_R = smb.r2t(ori) if is_R_valid(ori) else smb.r2t(make_R_valid(ori))
    R = sm.SE3(T_R, check=False).R

    # Combine translation and orientation
    T = sm.SE3.Rt(R=R, t=pos, check=False)

    return T
#============================================================================
def is_R_valid(R: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Checks if the input matrix R is a valid 3x3 rotation matrix.

    Parameters:
            R (np.ndarray): The input matrix to be checked.
            tol (float, optional): Tolerance for numerical comparison. Defaults to 1e-8.

    Returns:
            bool: True if R is a valid rotation matrix, False otherwise.

    Raises:
            ValueError: If R is not a 3x3 matrix.
    """
    # Check if R is a 3x3 matrix
    if not isinstance(R, np.ndarray) or R.shape != (3, 3):
        raise ValueError(f"Input is not a 3x3 matrix. R is \n{R}")

    # Check if R is orthogonal
    is_orthogonal = np.allclose(np.dot(R.T, R), np.eye(3), atol=tol)

    # Check if the determinant is 1
    det = np.linalg.det(R)

    return is_orthogonal and np.isclose(det, 1.0, atol=tol)
#=============================================================================
def make_R_valid(R: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Makes the input matrix R a valid 3x3 rotation matrix.

    Parameters:
            R (np.ndarray): The input matrix to be made valid.
            tol (float, optional): Tolerance for numerical comparison. Defaults to 1e-6.

    Returns:
            np.ndarray: A valid 3x3 rotation matrix derived from the input matrix R.

    Raises:
            ValueError: If the input is not a 3x3 matrix or if it cannot be made a valid rotation matrix.
    """
    if not isinstance(R, np.ndarray):
        R = np.array(R)

    # Check if R is a 3x3 matrix
    if R.shape != (3, 3):
        raise ValueError("Input is not a 3x3 matrix")

    # Step 1: Gram-Schmidt Orthogonalization
    Q, _ = np.linalg.qr(R)

    # Step 2: Ensure determinant is 1
    det = np.linalg.det(Q)
    if np.isclose(det, 0.0, atol=tol):
        raise ValueError("Invalid rotation matrix (determinant is zero)")

    # Step 3: Ensure determinant is positive
    if det < 0:
        Q[:, 2] *= -1

    return Q
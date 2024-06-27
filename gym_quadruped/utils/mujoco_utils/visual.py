import mujoco
import numpy as np
from mujoco.viewer import Handle
from scipy.spatial.transform import Rotation

from ..quadruped_utils import LegsAttr


def cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:  # See https://github.com/microsoft/pylance-release/issues/3277
    return np.cross(a, b)


def render_vector(viewer: Handle,
                  vector: np.ndarray,
                  pos: np.ndarray,
                  scale: float,
                  color: np.ndarray = np.array([1, 0, 0, 1]),
                  geom_id: int = -1) -> int:
    """
    Function to render a vector in the Mujoco viewer.

    Args:
        viewer (Handle): The Mujoco viewer.
        vector (np.ndarray): The vector to render.
        pos (np.ndarray): The position of the base of vector.
        scale (float): The scale of the vector.
        color (np.ndarray): The color of the vector.
        geom_id (int, optional): The id of the geometry. Defaults to -1.
    Returns:
        int: The id of the geometry.
    """
    if geom_id < 0:
        # Instantiate a new geometry
        geom = mujoco.MjvGeom()
        geom.type = mujoco.mjtGeom.mjGEOM_ARROW
        viewer.user_scn.ngeom += 1
        geom_id = viewer.user_scn.ngeom - 1

    geom = viewer.user_scn.geoms[geom_id]

    # Define the a rotation matrix with the Z axis aligned with the vector direction
    vec_z = vector.squeeze() / np.linalg.norm(vector + 1e-5)
    # Define any orthogonal to z vector as the X axis using the Gram-Schmidt process
    rand_vec = np.random.rand(3)
    vec_x = rand_vec - (np.dot(rand_vec, vec_z) * vec_z)
    vec_x = vec_x / np.linalg.norm(vec_x)
    # Define the Y axis as the cross product of X and Z
    vec_y = cross2(vec_z, vec_x)

    ori_mat = Rotation.from_matrix(np.array([vec_x, vec_y, vec_z]).T).as_matrix()
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.asarray([0.01, 0.01, scale]),
        pos=pos,
        mat=ori_mat.flatten(),
        rgba=color
        )

    return geom_id


def render_sphere(viewer: Handle,
                  position: np.ndarray,
                  diameter: float,
                  color: np.ndarray,
                  geom_id: int = -1) -> int:
    """
    Function to render a sphere in the Mujoco viewer.

    Args:
        viewer (Handle): The Mujoco viewer.
        position (np.ndarray): The position of the sphere.
        diameter (float): The diameter of the sphere.
        color (np.ndarray): The color of the sphere.
        geom_id (int, optional): The id of the geometry. Defaults to -1.
    Returns:
        int: The id of the geometry.
    """
    if geom_id < 0:
        # Instantiate a new geometry
        geom = mujoco.MjvGeom()
        geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
        viewer.user_scn.ngeom += 1
        geom_id = viewer.user_scn.ngeom - 1

    geom = viewer.user_scn.geoms[geom_id]

    # Initialize the geometry
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.asarray([diameter / 2] * 3),  # Radius is half the diameter
        mat=np.eye(3).flatten(),
        pos=position,
        rgba=color
        )

    return geom_id


def render_line(viewer, initial_point, target_point, width, color, geom_id=-1):
    """
    Function to render a line in the Mujoco viewer.

    Args:
        viewer (Handle): The Mujoco viewer.
        initial_point (np.ndarray): The initial point of the line.
        target_point (np.ndarray): The target point of the line.
        width (float): The width of the line.
        color (np.ndarray): The color of the line.
        geom_id (int, optional): The id of the geometry. Defaults to -1.
    Returns:
        int: The id of the geometry.
    """
    if geom_id < 0:
        # Instantiate a new geometry
        geom = mujoco.MjvGeom()
        viewer.user_scn.ngeom += 1
        geom_id = viewer.user_scn.ngeom - 1

    geom = viewer.user_scn.geoms[geom_id]

    # Define the rotation matrix with the Z axis aligned with the line direction
    vector = target_point - initial_point
    length = np.linalg.norm(vector)
    if length == 0:
        return geom_id

    vec_z = vector / length

    # Use Gram-Schmidt process to find an orthogonal vector for X axis
    rand_vec = np.random.rand(3)
    vec_x = rand_vec - np.dot(rand_vec, vec_z) * vec_z
    vec_x /= np.linalg.norm(vec_x)

    # Define the Y axis as the cross product of X and Z
    vec_y = cross2(vec_z, vec_x)

    ori_mat = Rotation.from_matrix(np.array([vec_x, vec_y, vec_z]).T).as_matrix()

    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=np.array([width, length / 2 + width / 4, width]),
        pos=(initial_point + target_point) / 2,
        mat=ori_mat.flatten(),
        rgba=color
        )

    return geom_id


def change_robot_appearance(mjModel: mujoco.MjModel, alpha=1.0):
    """Tint the robot in MuJoCo to get a similar visualization of symmetric robots."""
    # Define colors
    robot_color = [0.054, 0.415, 0.505, alpha]   # Awesome Teal
    FL_leg_color = [0.698, 0.376, 0.082, alpha]  # Awesome Orange
    FR_leg_color = [0.260, 0.263, 0.263, alpha]  # Awesome Grey
    HL_leg_color = [0.800, 0.480, 0.000, alpha]  # Awesome Yellow
    HR_leg_color = [0.710, 0.703, 0.703, alpha]  # Awesome Light grey

    for geom_id in range(mjModel.ngeom):
        geom_name = mujoco.mj_id2name(mjModel, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        prev_color = mjModel.geom_rgba[geom_id]
        body_id = mjModel.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(mjModel, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if geom_name in ['floor', 'plane', 'world', 'ground']: continue
        if prev_color[-1] == 0.0: continue  # Skip transparent geoms
        if body_name:
            # Determine the color based on the geom_name
            if any(s in body_name.lower() for s in ["fl_", "lf_", "left", "_0"]):
                color = FL_leg_color
            elif any(s in body_name.lower() for s in ["fr_", "rf_", "right", "_120"]):
                color = FR_leg_color
            elif any(s in body_name.lower() for s in ["rl_", "hl_", "lh_", "left"]):
                color = HL_leg_color
            elif any(s in body_name.lower() for s in ["rr_", "hr_", "rh_", "right"]):
                color = HR_leg_color
            else:
                color = robot_color

            # Change the visual appearance of the geom
            mjModel.geom_rgba[geom_id] = color


def render_ghost_robot(viewer, mjModel, mjData, alpha=0.5, ghost_geom_scn_ids=None):
    """Render a ghost robot in the MuJoCo viewer."""
    if ghost_geom_scn_ids is None:
        ghost_geom_scn_ids = {}

    # Iterate through all geometries in the model
    for model_geom_id in range(mjModel.ngeom):
        geom_name = mujoco.mj_id2name(mjModel, mujoco.mjtObj.mjOBJ_GEOM, model_geom_id)
        body_id = mjModel.geom_bodyid[model_geom_id]
        body_name = mujoco.mj_id2name(mjModel, mujoco.mjtObj.mjOBJ_BODY, body_id)
        geom_rgba = mjModel.geom_rgba[model_geom_id]

        # Rules to ignore some geometries we are not interested in rendering for visualization
        # Skip collision geometries (alpha == 0)
        if geom_name in ['floor', 'plane', 'world', 'ground']: continue
        if geom_rgba[3] == 0:
            continue

        # Create a name for the geometry (use body name if geom_name is None)
        ghost_geom_name = f"{model_geom_id}:{geom_name if geom_name is not None else body_name}"

        if ghost_geom_name not in ghost_geom_scn_ids:
            # Create a new scene geometry if it doesn't exist
            viewer.user_scn.ngeom += 1
            ghost_geom_scn_ids[ghost_geom_name] = viewer.user_scn.ngeom - 1

        # mjvGeom structure which will copy the mjModel geom properties
        ghost_geom = viewer.user_scn.geoms[ghost_geom_scn_ids[ghost_geom_name]]
        # Define the color with transparency
        color = geom_rgba.copy()
        color[3] = alpha
        # Update the geometry with the current position and orientation
        mujoco.mjv_initGeom(
            ghost_geom,
            type=mjModel.geom_type[model_geom_id],
            size=mjModel.geom_size[model_geom_id],
            pos=mjData.geom_xpos[model_geom_id],
            mat=mjData.geom_xmat[model_geom_id],
            rgba=color,
            category=mujoco.mjtCatBit.mjCAT_DECOR,   # This Geom should be treated as a decoration
            dataid=None,   # ?? mesh, hfield or plane id; -1: none
            objtype=None,  # ?? mujoco object type; mjOBJ_UNKNOWN for decor
            objid=None,    # ?? object id within the object type
            texid=None,    # ?? texture id; -1: no texture
            textuniform=None,  # ??  uniform cube mapping
            texcoord=None,  # ?? texture coordinates
            segid=-1,
            )

    return ghost_geom_scn_ids

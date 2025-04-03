from __future__ import annotations

import mujoco
import numpy as np
from mujoco.viewer import Handle
from scipy.spatial.transform import Rotation


def cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:  # See https://github.com/microsoft/pylance-release/issues/3277
    """."""
    return np.cross(a, b)


def render_vector(
    viewer: Handle,
    vector: np.ndarray,
    pos: np.ndarray,
    scale: float,
    color: np.ndarray = np.array([1, 0, 0, 1]),
    geom_id: int = -1,
) -> int:
    """Function to render a vector in the Mujoco viewer.

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
    if viewer is None:
        return -1
    if geom_id < 0:
        # Instantiate a new geometry
        geom = mujoco.MjvGeom()
        geom.type = mujoco.mjtGeom.mjGEOM_ARROW
        viewer.user_scn.ngeom += 1
        geom_id = viewer.user_scn.ngeom - 1

    geom = viewer.user_scn.geoms[geom_id]

    if np.isclose(np.linalg.norm(vector), 0, atol=1e-5):
        vector = np.random.rand(3)
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
        rgba=color,
    )
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    geom.segid = -1
    geom.objid = -1

    return geom_id


def render_sphere(viewer: Handle, position: np.ndarray, diameter: float, color: np.ndarray, geom_id: int = -1) -> int:
    """Function to render a sphere in the Mujoco viewer.

    Args:
        viewer (Handle): The Mujoco viewer.
        position (np.ndarray): The position of the sphere.
        diameter (float): The diameter of the sphere.
        color (np.ndarray): The color of the sphere.
        geom_id (int, optional): The id of the geometry. Defaults to -1.

    Returns:
        int: The id of the geometry.
    """
    if viewer is None:
        return -1

    if geom_id < 0 or geom_id is None:
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
        rgba=color,
    )

    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    geom.segid = -1
    geom.objid = -1

    return geom_id


def render_line(viewer: Handle, initial_point, target_point, width, color, geom_id=-1):
    """Function to render a line in the Mujoco viewer.

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
    if viewer is None:
        return -1
    if geom_id < 0 or geom_id is None:
        # Instantiate a new geometry
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
        rgba=color,
    )

    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    geom.segid = -1
    geom.objid = -1

    return geom_id


def change_robot_appearance(mjModel: mujoco.MjModel, alpha=1.0):
    """Tint the robot in MuJoCo to get a similar visualization of symmetric robots."""
    # Define colors
    robot_color = [0.054, 0.415, 0.505, alpha]  # Awesome Teal
    FL_leg_color = [0.698, 0.376, 0.082, alpha]  # Awesome Orange
    FR_leg_color = [0.260, 0.263, 0.263, alpha]  # Awesome Grey
    HL_leg_color = [0.800, 0.480, 0.000, alpha]  # Awesome Yellow
    HR_leg_color = [0.710, 0.703, 0.703, alpha]  # Awesome Light grey

    for geom_id in range(mjModel.ngeom):
        prev_color = mjModel.geom_rgba[geom_id]
        body_id = mjModel.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(mjModel, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name in ['floor', 'plane', 'world', 'ground']:
            continue
        if prev_color[-1] == 0.0:
            continue  # Skip transparent geoms
        if body_name:
            # Determine the color based on the geom_name
            if any(s in body_name.lower() for s in ['fl_', 'lf_', 'left', '_0']):
                color = FL_leg_color
            elif any(s in body_name.lower() for s in ['fr_', 'rf_', 'right', '_120']):
                color = FR_leg_color
            elif any(s in body_name.lower() for s in ['rl_', 'hl_', 'lh_', 'left']):
                color = HL_leg_color
            elif any(s in body_name.lower() for s in ['rr_', 'hr_', 'rh_', 'right']):
                color = HR_leg_color
            else:
                color = robot_color
            # print(f'{geom_name} {body_name}: {color}')
            # Change the visual appearance of the geom
            mjModel.geom_rgba[geom_id] = color


def render_ghost_robot(
    viewer: Handle, mj_model: mujoco.MjModel, mj_data: mujoco.MjData, alpha=0.5, ghost_geoms: dict | None = None
):
    """Render a ghost robot in the viewer with transparency.

    :param viewer: Mujoco Handle to the viewer. Assumed to be in passive mode.
    :param mj_model: Mujoco MjModel containing the Geoms of the robot.
    :param mj_data: Mujoco MjData containing the position (xpos) and orientation (xmat) of the Geoms.
    :param alpha: The transparency of the ghost robot. 0.0 is fully transparent, 1.0 is fully opaque.
    :param ghost_geoms: A dictionary with keys as the decorative geometry ids (idx in `viewer.user_scn.geoms`) and
     values as the corresponding `MjvGeom` from visual Geometries of the MjModel. If None, the function will
     automatically create the ghost_geoms and return this dictionary so recurrent calls avoid re-creating the
     ghost_geoms.
    :return: A dictionary with keys as the decorative geometry ids (idx in `viewer.user_scn.geoms`) and values as the
            corresponding `MjvGeom` from visual Geometries of the MjModel.
    """
    if ghost_geoms is None or len(ghost_geoms) == 0:
        # Hacky way to get the robot Geometries from a "passive" viewer, as we cant get these from the viewer.user_scn
        # See: https://github.com/google-deepmind/mujoco/issues/1757
        scene = mujoco.MjvScene(mj_model, 200)
        mujoco.mjv_updateScene(
            mj_model,
            mj_data,  # Create mock scene to get the Geoms
            mujoco.MjvOption(),
            None,
            mujoco.MjvCamera(),
            mujoco.mjtCatBit.mjCAT_ALL,
            scene,
        )
        visible_geoms = [g for g in scene.geoms[: scene.ngeom] if g.segid != -1]  # All non-decorative geoms in scene
        visible_objid = np.array([g.objid for g in visible_geoms], np.int32)  # ObjectID = model_geom_id
        visible_objtype = np.array([g.objtype for g in visible_geoms], np.int32)

        ghost_geoms = {}
        for geom, model_geom_id, _objtype in zip(visible_geoms, visible_objid, visible_objtype, strict=False):
            geom_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, model_geom_id)
            body_id = mj_model.geom_bodyid[model_geom_id]
            body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            geom_rgba = mj_model.geom_rgba[model_geom_id]

            # Rule set of geoms to ignore. We dont want collision geoms or the floor ______________________________
            ignored_names = ['floor', 'plane', 'world', 'ground']
            if geom_name in ignored_names or body_name in ignored_names or geom_rgba[3] == 0:
                continue
            # ________________________________________________________________________________________________________

            viewer.user_scn.ngeom += 1
            dec_geom_scn_id = viewer.user_scn.ngeom - 1
            ghost_geoms[dec_geom_scn_id] = geom  # Store a map from `viewer.user_scn.geom_id` and model's `MjvGeom`

    for dec_geom_scn_id, geom in ghost_geoms.items():
        geom_model_id = geom.objid
        # Update the color of the geometry with transparency
        geom.rgba[3] = alpha
        # Copy the Body Geom to the decorative geom in viewer.user_scn
        dec_geom = viewer.user_scn.geoms[dec_geom_scn_id]
        mujoco.mjv_initGeom(
            dec_geom,
            type=geom.type,
            rgba=geom.rgba,
            size=geom.size,
            # Use the position and orientation from the mjData
            pos=mj_data.geom_xpos[geom_model_id],
            mat=mj_data.geom_xmat[geom_model_id].reshape(9),
        )
        # Ensure ghost decorative geometries are ignored in segmentation, collisions, etc.
        dec_geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        dec_geom.segid = -1
        dec_geom.objid = -1
        dec_geom.reflectance = 0.0
        # Copy the mesh and texture attributes
        dec_geom.dataid = geom.dataid
        # dec_geom.texid = geom.texid
        # dec_geom.texcoord = geom.texcoord
        # dec_geom.texrepeat = geom.texrepeat
        # dec_geom.texuniform = geom.texuniform
        dec_geom.emission = geom.emission
        dec_geom.specular = geom.specular
        dec_geom.shininess = geom.shininess

    return ghost_geoms

from __future__ import annotations

import itertools
import os
import time
import warnings
from collections import namedtuple
from pathlib import Path
from typing import Any, Union

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces
from mujoco import MjData, MjModel
from scipy.spatial.transform import Rotation

from gym_quadruped.utils.math_utils import homogenous_transform
from gym_quadruped.utils.mujoco.visual import change_robot_appearance, render_ghost_robot, render_vector
from gym_quadruped.utils.quadruped_utils import LegsAttr, extract_mj_joint_info

BASE_OBS = ['base_pos',
            'base_lin_vel', 'base_lin_vel:base',
            'base_lin_acc', 'base_lin_acc:base',
            'base_ang_vel', 'base_ang_vel:base',
            'base_ori_euler_xyz', 'base_ori_quat_wxyz', 'base_ori_SO3']
GEN_COORDS_OBS = ['qpos', 'qvel', 'tau_ctrl_setpoint', 'qpos_js', 'qvel_js']
FEET_OBS = ['feet_pos', 'feet_pos:base', 'feet_vel', 'feet_vel:base', 'contact_state', 'contact_forces',
            'contact_forces:base']


class QuadrupedEnv(gym.Env):
    """A simple quadruped environment for testing model-based controllers and imitation learning algorithms.

    To deal with different quadruped robots, which might have different joint naming and ordering conventions, this
    environment uses the `LegsAttr` dataclass to store attributes associated with the legs of a quadruped robot. This
    dataclass uses the naming convention FL, FR, RL, RR to represent the Front-Left, Front-Right, Rear-Right, and
    Rear-Left legs, respectively.
    """

    _DEFAULT_OBS = ('qpos', 'qvel', 'tau_ctrl_setpoint', 'feet_pos_base', 'feet_vel_base')
    ALL_OBS = BASE_OBS + GEN_COORDS_OBS + FEET_OBS

    metadata = {'render.modes': ['human'], 'version': 0}

    def __init__(self,
                 robot: str,
                 hip_height: float,
                 legs_joint_names: dict,
                 state_obs_names: tuple[str, ...] = _DEFAULT_OBS,
                 scene: str = 'flat',
                 sim_dt: float = 0.002,
                 base_vel_command_type: str = 'forward',
                 ref_base_lin_vel: Union[tuple[float, float], float] = 0.5,  # [m/s]
                 ref_base_ang_vel: Union[tuple[float, float], float] = 0.0,  # [rad/s]
                 ground_friction_coeff: Union[tuple[float, float], float] = 1.0,
                 legs_order: tuple[str, str, str, str] = ('FL', 'FR', 'RL', 'RR'),
                 feet_geom_name: dict = None,
                 ):
        """Initialize the quadruped environment.

        Args:
        ----
            robot: Robot model name.
            hip_height: Nominal height of the robot's hip from the ground.
            legs_joint_names: (dict) Dict with keys FL, FR, RL, RR; and as values a list of joint names associated with
                each of the four legs.
            state_obs_names: (tuple) Names of the state observations to include in the observation space.
            scene: Name of the ground terrain, available are 'flat', 'perlin', TODO: more
            sim_dt: Time step of the mujoco simulation.
            base_vel_command_type: Type of base velocity command, options are:
                - 'forward': The velocity commands are in the "forward" direction of the robot.
                - 'random': The velocity commands are in random "heading" directions.
                - 'human': The velocity commands are defined by the arrow keys of the keyboard.
                    - Up arrow: increase forward velocity.
                    - Down arrow: decrease forward velocity.
                    - Left arrow: increase base angular (d yaw / dt) velocity.
                    - Right arrow: decrease base angular (d yaw / dt) velocity.
            ref_base_lin_vel: Magnitude of the desired/reference base linear velocity command. If a float, the
                velocity command is fixed. If a tuple, (min, max) the velocity command is uniformly sampled from this
            ref_base_ang_vel: Magnitude of the desired/reference base angular velocity (d yaw / dt) command.
                If a float, the velocity command is fixed. If a tuple, (min, max) the velocity command is uniformly
            ground_friction_coeff: Magnitude of the ground lateral friction coefficient. If a float, the friction
                coefficient is fixed. If a tuple, (min, max) the friction coefficient is uniformly sampled from this
            legs_order: (tuple) Default order of the legs in the state observation and action space. This order defines
                how state observations of legs observables (e.g., feet positions) are ordered in the state vector.
            feet_geom_name: (dict) Dict with keys FL, FR, RL, RR; and as values the name of the Mujoco geometry
                (MjvGeom) associated with the feet/contact-point of each leg. Used to compute the ground contact forces.
        """
        super(QuadrupedEnv, self).__init__()

        # Store all initialization arguments in a dictionary. Useful if we want to reconstruct this environment.
        self._save_hyperparameters(constructor_params=locals().copy())

        self.hip_height = hip_height
        self.base_vel_command_type = base_vel_command_type
        self.base_lin_vel_range = ref_base_lin_vel if isinstance(ref_base_lin_vel, tuple) else (ref_base_lin_vel,
                                                                                                ref_base_lin_vel)
        self.base_ang_vel_range = ref_base_ang_vel if isinstance(ref_base_ang_vel, tuple) else (ref_base_ang_vel,
                                                                                                ref_base_ang_vel)
        self.ground_friction_coeff_range = ground_friction_coeff if isinstance(ground_friction_coeff, tuple) else (
            ground_friction_coeff, ground_friction_coeff)
        self.legs_order = legs_order

        # Define the model and data _______________________________________________________________
        # TODO: We should create a scene with the desired terrain and load the robot model (or multiple instances of the
        #   robot models relying on robot_descriptions.py. This way of loading the XML is not ideal.
        dir_path = os.path.dirname(os.path.realpath(__file__))
        base_path = Path(dir_path) / 'robot_model' / robot
        model_file_path = base_path / f'scene_{scene}.xml'
        print(model_file_path)
        assert model_file_path.exists(), f"Model file not found: {model_file_path.absolute().resolve()}"

        # Load the robot and scene to mujoco
        try:
            self.mjModel: MjModel = mujoco.MjModel.from_xml_path(str(model_file_path))
        except ValueError as e:
            raise ValueError(f"Error loading the scene {model_file_path}:") from e

        self.mjData: MjData = mujoco.MjData(self.mjModel)
        # MjData structure to compute and store the state of a ghost/transparent robot for visual rendering.
        self._ghost_mjData: MjData = mujoco.MjData(self.mjModel)

        # Set the simulation step size (dt)
        self.mjModel.opt.timestep = sim_dt

        # Identify the legs DoF indices/address in the qpos and qvel arrays ___________________________________________
        assert legs_joint_names is not None, "Please provide the joint names associated with each of the four legs."
        self.joint_info = extract_mj_joint_info(self.mjModel)
        self.legs_qpos_idx = LegsAttr(None, None, None, None)  # Indices of legs joints in qpos vector
        self.legs_qvel_idx = LegsAttr(None, None, None, None)  # Indices of legs joints in qvel vector
        self.legs_tau_idx = LegsAttr(None, None, None, None)   # Indices of legs actuators in gen forces vector
        # Ensure the joint names of the robot's legs' joints are in the model. And store the qpos and qvel indices
        for leg_name in ["FR", "FL", "RR", "RL"]:
            qpos_idx, qvel_idx, tau_idx = [], [], []
            leg_joints = legs_joint_names[leg_name]
            for joint_name in leg_joints:
                assert joint_name in self.joint_info, f"Joint {joint_name} not found in {list(self.joint_info.keys())}"
                qpos_idx.extend(self.joint_info[joint_name].qpos_idx)
                qvel_idx.extend(self.joint_info[joint_name].qvel_idx)
                tau_idx.extend(self.joint_info[joint_name].tau_idx)
            self.legs_qpos_idx[leg_name] = qpos_idx
            self.legs_qvel_idx[leg_name] = qvel_idx
            self.legs_tau_idx[leg_name] = tau_idx

        # If the feet geometry names are provided, store the geom ids and body ids of the feet geometries _____________
        self._feet_geom_id, self._feet_body_id = LegsAttr(None, None, None, None), LegsAttr(None, None, None, None)
        if feet_geom_name is not None:
            self._find_feet_model_attrs(feet_geom_name)

        # Action space: Torque values for each joint _________________________________________________________________
        tau_low, tau_high = self.mjModel.actuator_forcerange[:, 0], self.mjModel.actuator_forcerange[:, 1]
        is_act_lim = [np.inf if not lim else 1.0 for lim in self.mjModel.actuator_forcelimited]
        self.action_space = spaces.Box(
            shape=(self.mjModel.nu,),
            low=np.asarray([tau if not lim else -np.inf for tau, lim in zip(tau_low, is_act_lim)]),
            high=np.asarray([tau if not lim else np.inf for tau, lim in zip(tau_high, is_act_lim)]),
            dtype=np.float32)

        # Observation space: __________________________________________________________________________________________
        # Get the Env observation gym.Space, and a dict with the indices of each observation in the state vector
        self.observation_space, self.state_obs_idx = self._configure_observation_space(state_obs_names)
        self.state_obs_names = state_obs_names

        self.viewer = None
        self.step_num = 0
        # Reference base velocity in "Horizontal" frame (see heading_orientation_SO3)
        self._ref_base_lin_vel_H, self._ref_base_ang_yaw_dot = None, None
        # Store the ids of visual aid geometries
        self._geom_ids, self._ghost_robots_geom = {}, {}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply the action to the robot, evolve the simulation, and return the observation, reward, and termination.

        Args:
        ----
            action: (np.ndarray) The desired joint-space torques to apply to each of the robot's DoF actuators.

        Returns:
        -------
            np.ndarray: The state observation.
            float: The reward.
            bool: Whether the episode is terminated.
            bool: Whether the episode is truncated.
            dict: Additional information.
        """
        # Apply action (torque) to the robot
        self.mjData.ctrl = action
        mujoco.mj_step(self.mjModel, self.mjData)

        # Get observation
        obs = self._get_obs()

        # Compute reward (simplified)
        reward = self._compute_reward()

        # Check if done (simplified, usually more complex)
        invalid_contact, contact_info = self._check_for_invalid_contacts()
        is_terminated = invalid_contact  # and ...
        is_truncated = False
        # Info dictionary
        info = dict(time=self.mjData.time, step_num=self.step_num, invalid_contacts=contact_info)

        self.step_num += 1
        return obs, reward, is_terminated, is_truncated, info

    def reset(self,
              qpos: np.ndarray = None,
              qvel: np.ndarray = None,
              seed: int | None = None,
              random: bool = True,
              options: dict[str, Any] | None = None) -> np.ndarray:
        """Reset the environment.

        Args:
        ----
            qpos: (np.ndarray) Initial joint positions. If None, random initialization around keyframe 0.
            qvel: (np.ndarray) Initial joint velocities. If None, random initialization around keyframe 0.
            seed: (int) Seed for reproducibility.
            random: (bool) Whether to randomize the initial state.
            options: (dict) Additional options for the reset.

        Returns:
        -------
            np.ndarray: The initial observation.
        """
        # Reset relevant variables
        self.step_num = 0
        self.mjData.time = 0.0
        self.mjData.ctrl = 0.0  # Reset control signals
        self.mjData.qfrc_applied = 0.0
        options = {} if options is None else options

        if seed is not None: np.random.seed(seed)  # Set seed for reproducibility

        # Reset the robot state ----------------------------------------------------------------------------------------
        if qpos is None and qvel is None:  # Random initialization around xml keyframe 0
            mujoco.mj_resetDataKeyframe(self.mjModel, self.mjData, 0)
            # Add white noise to the joint-space position and velocity
            if random:
                q_pos_amp = 25 * np.pi / 180 if 'angle_sweep' not in options else options['angle_sweep']
                q_vel_amp = 0.1
                self.mjData.qpos[7:] += np.random.uniform(-q_pos_amp, q_pos_amp, self.mjModel.nq - 7)
                self.mjData.qvel[6:] += np.random.uniform(-q_vel_amp, q_vel_amp, self.mjModel.nv - 6)
                # Random orientation
                roll_sweep = 10 * np.pi / 180 if 'roll_sweep' not in options else options['roll_sweep']
                pitch_sweep = 10 * np.pi / 180 if 'pitch_sweep' not in options else options['pitch_sweep']
                ori_xyzw = Rotation.from_euler('xyz',
                                               [np.random.uniform(-roll_sweep, roll_sweep),
                                                np.random.uniform(-pitch_sweep, pitch_sweep),
                                                np.random.uniform(-np.pi, np.pi)]).as_quat(canonical=True)
                ori_wxyz = np.roll(ori_xyzw, 1)
                self.mjData.qpos[3:7] = ori_wxyz
                # Random xy position withing a 2 x 2 square
                self.mjData.qpos[0:2] = np.random.uniform(-2, 2, 2)

                try:
                    feet_pos = self.feet_pos(frame='world')
                    max_z = np.max([pos[2] for pos in feet_pos.to_list()])
                    self.mjData.qpos[2] -= max_z
                except ValueError as e:
                    self.mjData.qpos[2] = self.hip_height + np.random.uniform(-0.05 * self.hip_height,
                                                                              0.05 * self.hip_height)


            # Perform a forward dynamics computation to update the contact information
            mujoco.mj_step1(self.mjModel, self.mjData)
            # Check if the robot is in contact with the ground. If true lift the robot until contact is broken.
            contact_state, contacts = self.feet_contact_state()
            while np.any(contact_state.to_list()):
                all_contacts = list(itertools.chain(*contacts.to_list()))
                max_penetration_distance = np.max([np.abs(contact.dist) for contact in all_contacts])
                self.mjData.qpos[2] += max_penetration_distance * 1.1
                mujoco.mj_step1(self.mjModel, self.mjData)
                contact_state, contacts = self.feet_contact_state()
        else:
            self.mjData.qpos = qpos
            self.mjData.qvel = qvel

        # Reset the accelerations to zero
        self.mjData.qacc[:] = 0
        self.mjData.qacc_warmstart[:] = 0
        # This ensures all registers/arrays are updated
        mujoco.mj_step(self.mjModel, self.mjData)

        # Reset the desired base velocity command
        # ----------------------------------------------------------------------
        if 'forward' in self.base_vel_command_type:
            base_vel_norm = np.random.uniform(*self.base_lin_vel_range)
            base_heading_vel_vec = np.array([1, 0, 0])  # Move in the "forward" direction
        elif 'random' in self.base_vel_command_type:
            base_vel_norm = np.random.uniform(*self.base_lin_vel_range)
            heading_angle = np.random.uniform(-np.pi, np.pi)
            base_heading_vel_vec = np.array([np.cos(heading_angle), np.sin(heading_angle), 0])
        elif 'human' in self.base_vel_command_type:
            base_vel_norm = 0.0
            base_heading_vel_vec = np.array([1, 0, 0])
            self._ref_base_ang_yaw_dot = 0.0
        else:
            raise ValueError(f"Invalid base linear velocity command type: {self.base_vel_command_type}")

        if 'rotate' in self.base_vel_command_type:
            self._ref_base_ang_yaw_dot = np.random.uniform(*self.base_ang_vel_range)
        else:
            self._ref_base_ang_yaw_dot = 0.0

        self._ref_base_lin_vel_H = base_vel_norm * base_heading_vel_vec

        # Ground friction coefficient randomization if enabled.
        tangential_friction = np.random.uniform(*self.ground_friction_coeff_range)
        self._set_ground_friction(tangential_coeff=tangential_friction)

        return self._get_obs()

    def render(self, mode='human', tint_robot=False, ghost_qpos=None, ghost_alpha=0.5):
        """ Render the environment.

        Args:
            mode: (str) The rendering mode. Only 'human' is supported. TODO: rgb frame.
            tint_robot: (bool) Whether to tint the robot with a color.
            ghost_qpos: (nq,) or (n_robot, nq) array with the joint positions of the ghost robots.
            ghost_alpha: (float) or (n_robot,) array of alpha value of the ghost robots.
        Returns:

        """
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(
                self.mjModel, self.mjData, show_left_ui=False, show_right_ui=False,
                key_callback=lambda x: self._key_callback(x)
                )
            if tint_robot: change_robot_appearance(self.mjModel, alpha=1.0)

            mujoco.mjv_defaultFreeCamera(self.mjModel, self.viewer.cam)
            self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0

        # Define/Update markers for visualization of desired and current base velocity _______________________________
        X_B = self.base_configuration
        base_pos = X_B[:3, 3]
        base_lin_vel = self.mjData.qvel[0:3]
        ref_base_lin_vel_B, _ = self.target_base_vel()
        ref_vec_pos, vec_pos = base_pos + [0, 0, 0.1], base_pos + [0, 0, 0.15]
        ref_vel_vec_color, vel_vec_color = np.array([1, 0.5, 0, .7]), np.array([0, 1, 1, .7])
        ref_vec_scale, vec_scale = np.linalg.norm(ref_base_lin_vel_B) / 1.0, np.linalg.norm(base_lin_vel) / 1.0

        ref_vec_id, vec_id = self._geom_ids.get('ref_dr_B_vec', -1), self._geom_ids.get('dr_B_vec', -1)
        self._geom_ids['ref_dr_B_vec'] = render_vector(
            self.viewer, ref_base_lin_vel_B, pos=ref_vec_pos, scale=ref_vec_scale, color=ref_vel_vec_color,
            geom_id=ref_vec_id
            )
        self._geom_ids['dr_B_vec'] = render_vector(
            self.viewer, base_lin_vel, pos=vec_pos, scale=vec_scale, color=vel_vec_color,
            geom_id=vec_id
            )

        # Ghost robot rendering _______________________________________________________________________________________
        if ghost_qpos is not None:
            self._render_ghost_robots(qpos=ghost_qpos, alpha=ghost_alpha)

        # Update the camera position. _________________________________________________________________________________
        cam_pos = max(self.hip_height * 0.1, base_pos[2])
        self._update_camera_target(self.viewer.cam, np.concatenate((base_pos[:2], [cam_pos])))

        # Finally, sync the viewer with the data. # TODO: if render mode is rgb, return the frame.
        self.viewer.sync()

    def target_base_vel(self):
        """Returns the target base linear (3,) and angular (3,) velocity in the world reference frame."""
        if self._ref_base_lin_vel_H is None:
            raise RuntimeError("Please call env.reset() before accessing the target base velocity.")
        R_B_heading = self.heading_orientation_SO3
        ref_base_lin_vel = (R_B_heading @ self._ref_base_lin_vel_H.reshape(3, 1)).squeeze()
        ref_base_ang_vel = np.array([0., 0., self._ref_base_ang_yaw_dot])
        return ref_base_lin_vel, ref_base_ang_vel

    def base_lin_vel(self, frame='world'):
        """Returns the base linear velocity (3,) in the specified frame."""
        if frame == 'world':
            return self.mjData.qvel[0:3]
        elif frame == 'base':
            R = self.base_configuration[0:3, 0:3]
            return R.T @ self.mjData.qvel[0:3]
        else:
            raise ValueError(f"Invalid frame: {frame} != 'world' or 'base'")

    def base_ang_vel(self, frame='world'):
        """Returns the base angular velocity (3,) in the specified frame."""
        if frame == 'world':
            return self.mjData.qvel[3:6]
        elif frame == 'base':
            R = self.base_configuration[0:3, 0:3]
            return R.T @ self.mjData.qvel[3:6]
        else:
            raise ValueError(f"Invalid frame: {frame} != 'world' or 'base'")
        
    def base_lin_acc(self, frame='world'):
        """Returns the base linear acceleration (3,) [m/s^2] in the specified frame."""
        if frame == 'world':
            return self.mjData.qacc[0:3]
        elif frame == 'base':
            R = self.base_configuration[0:3, 0:3]
            return R.T @ self.mjData.qacc[0:3]
        else:
            raise ValueError(f"Invalid frame: {frame} != 'world' or 'base'")

    def get_base_inertia(self) -> np.ndarray:
        """Returns the reflected rotational inertia of the robot's base at the current configuration in world frame.

        Args:
        ----
            model: The MuJoCo model.
            data: The MuJoCo data.

        Returns:
        -------
            np.ndarray: The reflected rotational inertia matrix in the world frame.
        """
        # Initialize the full mass matrix
        mass_matrix = np.zeros((self.mjModel.nv, self.mjModel.nv))
        mujoco.mj_fullM(self.mjModel, mass_matrix, self.mjData.qM)

        # Extract the 3x3 rotational inertia matrix of the base (assuming the base has 6 DoFs)
        inertia_B_at_qpos = mass_matrix[3:6, 3:6]

        return inertia_B_at_qpos

    def hip_positions(self, frame='world') -> LegsAttr:
        """Get the hip positions in the specified frame.

        Args:
        ----
            frame:  Either 'world' or 'base'. The reference frame in which the hip positions are computed.

        Returns:
        -------
            LegsAttr: A dictionary-like object with:
                - FR: (3,) position of the FR hip in the specified frame.
                - FL: (3,) position of the FL hip in the specified frame.
                - RL: (3,) position of the RL hip in the specified frame.
                - RR: (3,) position of the RR hip in the specified frame.
        """
        if frame == 'world':
            R = np.eye(3)
        elif frame == 'base':
            R = self.base_configuration[0:3, 0:3]
        else:
            raise ValueError(f"Invalid frame: {frame} != 'world' or 'base'")
        # TODO: Name of bodies should not be hardcodd
        FL_hip_id = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_BODY, 'FL_hip')
        FR_hip_id = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_BODY, 'FR_hip')
        RL_hip_id = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_BODY, 'RL_hip')
        RR_hip_id = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_BODY, 'RR_hip')
        return LegsAttr(
            FR=R.T @ self.mjData.body(FR_hip_id).xpos,
            FL=R.T @ self.mjData.body(FL_hip_id).xpos,
            RR=R.T @ self.mjData.body(RR_hip_id).xpos,
            RL=R.T @ self.mjData.body(RL_hip_id).xpos,
            )

    def feet_pos(self, frame='world') -> LegsAttr:
        """Get the feet positions in the specified frame.

        Args:
        ----
            frame: Either 'world' or 'base'. The reference frame in which the feet positions are computed.

        Returns:
        -------
            LegsAttr: A dictionary-like object with:
                - FR: (3,) position of the FR foot in the specified frame.
                - FL: (3,) position of the FL foot in the specified frame.
                - RR: (3,) position of the RR foot in the specified frame.
                - RL: (3,) position of the RL foot in the specified frame.
        """
        if any(x is None for x in self._feet_geom_id.to_list()):
            raise ValueError(
                "Please provide the `feet_geom_name` argument in the Env constructor to compute feet positions."
                )

        if frame == 'world':
            X = np.eye(4)
        elif frame == 'base':
            X = np.linalg.inv(self.base_configuration)  # X_W2B : World to Base
        else:
            raise ValueError(f"Invalid frame: {frame} != 'world' or 'base'")

        return LegsAttr(
            FR=homogenous_transform(self.mjData.geom_xpos[self._feet_geom_id.FR], X),
            FL=homogenous_transform(self.mjData.geom_xpos[self._feet_geom_id.FL], X),
            RR=homogenous_transform(self.mjData.geom_xpos[self._feet_geom_id.RR], X),
            RL=homogenous_transform(self.mjData.geom_xpos[self._feet_geom_id.RL], X),
            )

    def feet_vel(self, frame='world') -> LegsAttr:
        """Get the feet velocities in the specified frame.

        The feet velocities are computed using the Jacobians of the feet positions and the joint velocities.

        Args:
        ----
            frame: Either 'world' or 'base'. The reference frame in which the feet velocities are computed.

        Returns:
        -------
            LegsAttr: A dictionary-like object with:
                - FR: (3,) velocity of the FR foot in the specified frame.
                - FL: (3,) velocity of the FL foot in the specified frame.
                - RR: (3,) velocity of the RR foot in the specified frame.
                - RL: (3,) velocity of the RL foot in the specified frame.
        """
        feet_jac = self.feet_jacobians(frame=frame)
        return LegsAttr(**{leg_name: feet_jac[leg_name] @ self.mjData.qvel for leg_name in self.legs_order})

    def feet_jacobians(self, frame: str = 'world', return_rot_jac: bool = False) -> LegsAttr | tuple[LegsAttr, ...]:
        """Compute the Jacobians of the feet positions.

        This function computes the translational and rotational Jacobians of the feet positions. Each feet position is
        defined as the position of the geometry corresponding to each foot, passed in the `feet_geom_name` argument of
        the constructor. The body to which each feet point/geometry is attached to is assumed to be the one passed in
        the `feet_body_name` argument of the constructor.

        The Jacobians returned can be used to compute the relationship between joint velocities and feet velocities,
        such that if r_dot_FL is the velocity of the FL foot in the world frame, then:
        r_dot_FL = J_FL @ qvel, where J_FL in R^{3 x mjModel.nv} is the Jacobian of the FL foot position.

        Args:
        ----
            frame: Either 'world' or 'base'. The reference frame in which the Jacobians are computed.
            return_rot_jac: Whether to compute the rotational Jacobians. If False, only the translational Jacobians
                are computed.

        Returns:
        -------
            If `return_rot_jac` is False:
            LegsAttr: A dictionary-like object with:
                - FR: (3, mjModel.nv) Jacobian of the FR foot position in the specified frame.
                - FL: (3, mjModel.nv) Jacobian of the FL foot position in the specified frame.
                - RR: (3, mjModel.nv) Jacobian of the RR foot position in the specified frame.
                - RL: (3, mjModel.nv) Jacobian of the RL foot position in the specified frame.
            If `return_rot_jac` is True:
            tuple: A tuple with two LegsAttr objects:
                - The first LegsAttr object contains the translational Jacobians as described above.
                - The second LegsAttr object contains the rotational Jacobians.
        """
        if any(x is None for x in self._feet_body_id.to_list()):
            raise ValueError(
                "Please provide the `feet_geom_name` argument in the Env constructor to compute feet Jacobians."
                )

        if frame == 'world':
            R = np.eye(3)
        elif frame == 'base':
            R = self.base_configuration[0:3, 0:3]
        else:
            raise ValueError(f"Invalid frame: {frame} != 'world' or 'base'")
        feet_trans_jac = LegsAttr(*[np.zeros((3, self.mjModel.nv)) for _ in range(4)])
        feet_rot_jac = LegsAttr(*[np.zeros((3, self.mjModel.nv)) if not return_rot_jac else None for _ in range(4)])
        feet_pos = self.feet_pos(frame='world')  # Mujoco mj_jac expects the point in global coordinates.

        for leg_name in ["FR", "FL", "RR", "RL"]:
            mujoco.mj_jac(m=self.mjModel,
                          d=self.mjData,
                          jacp=feet_trans_jac[leg_name],
                          jacr=feet_rot_jac[leg_name],
                          point=feet_pos[leg_name],  # Point in global coordinates
                          body=self._feet_body_id[leg_name]  # Body to which `point` is attached to.
                          )
            feet_trans_jac[leg_name] = R.T @ feet_trans_jac[leg_name]
            if return_rot_jac:
                feet_rot_jac[leg_name] = R.T @ feet_rot_jac[leg_name]

        return feet_trans_jac if not return_rot_jac else (feet_trans_jac, feet_rot_jac)

    def feet_contact_state(self, frame='world', ground_reaction_forces=False) -> [LegsAttr, LegsAttr]:
        """Returns the boolean contact state of the feet.

        This function considers only contacts between the feet and the ground.

        Args:
        ----
            frame: (str) The reference frame in which the contact forces are computed. Either 'world' or 'base'.
            ground_reaction_forces: (bool) Whether to compute the total ground reaction forces for each foot.

        Returns:
        -------
            LegsAttr: A dictionary-like object with:
                - FL: (bool) True if the FL foot is in contact with the ground.
                - FR: (bool) True if the FR foot is in contact with the ground.
                - RL: (bool) True if the RL foot is in contact with the ground.
                - RR: (bool) True if the RR foot is in contact with the ground.
            LegsAttr: A dictionary-like object with:
                - FL: list[MjContact] A list of contact objects associated with the FL foot.
                - FR: list[MjContact] A list of contact objects associated with the FR foot.
                - RL: list[MjContact] A list of contact objects associated with the RL foot.
                - RR: list[MjContact] A list of contact objects associated with the RR foot.
            if ground_reaction_forces is True:
                LegsAttr: A dictionary-like object with:
                    - FL: (3,) The total ground reaction force acting on the FL foot in the specified frame.
                    - FR: (3,) The total ground reaction force acting on the FR foot in the specified frame.
                    - RL: (3,) The total ground reaction force acting on the RL foot in the specified frame.
                    - RR: (3,) The total ground reaction force acting on the RR foot in the specified frame.
        """

        if any(x is None for x in self._feet_body_id.to_list()):
            raise ValueError(
                "Please provide the `feet_geom_name` argument in the Env constructor to compute contact forces."
                )

        contact_state = LegsAttr(FL=False, FR=False, RL=False, RR=False)
        feet_contacts = LegsAttr(FL=[], FR=[], RL=[], RR=[])
        feet_contact_forces = LegsAttr(FL=[], FR=[], RL=[], RR=[])
        for contact_id, contact in enumerate(self.mjData.contact):
            # Get body IDs from geom IDs
            body1_id = self.mjModel.geom_bodyid[contact.geom1]
            body2_id = self.mjModel.geom_bodyid[contact.geom2]

            if 0 in [body1_id, body2_id]:  # World body ID is 0
                second_id = body2_id if body1_id == 0 else body1_id
                if second_id in self._feet_body_id.to_list():  # Check if contact occurs with the feet
                    for leg_name in ["FL", "FR", "RL", "RR"]:
                        if second_id == self._feet_body_id[leg_name]:
                            contact_state[leg_name] = True
                            feet_contacts[leg_name].append(contact)
                            if ground_reaction_forces:  # Store the contact forces
                                # Contact normal is R_c[:,0], that is the x-axis of the contact frame
                                R_c = contact.frame.reshape(3, 3)
                                force_c = np.zeros(6)  # 6D wrench vector
                                mujoco.mj_contactForce(self.mjModel, self.mjData, id=contact_id, result=force_c)
                                # Transform the contact force to the world frame
                                force_w = R_c.T @ force_c[:3]
                                feet_contact_forces[leg_name].append(force_w)

        if ground_reaction_forces:
            if frame == 'world':
                R = np.eye(3)
            elif frame == 'base':
                R = self.base_configuration[0:3, 0:3]
            else:
                raise ValueError(f"Invalid frame: {frame} != 'world' or 'base'")
            # Compute the total ground reaction force for each foot
            for leg_name in ["FL", "FR", "RL", "RR"]:
                if contact_state[leg_name]:
                    feet_contact_forces[leg_name] = R.T @ np.sum(feet_contact_forces[leg_name], axis=0)
                else:
                    feet_contact_forces[leg_name] = np.zeros(3)
            return contact_state, feet_contacts, feet_contact_forces

        return contact_state, feet_contacts

    def close(self):
        """Close the viewer."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    @property
    def legs_mass_matrix(self, ):
        mass_matrix = np.zeros((self.mjModel.nv, self.mjModel.nv))
        mujoco.mj_fullM(self.mjModel, mass_matrix, self.mjData.qM)
        # Get the mass matrix of the legs
        legs_mass_matrix = LegsAttr(FL=mass_matrix[np.ix_(self.legs_qvel_idx.FL, self.legs_qvel_idx.FL)],
                                    FR=mass_matrix[np.ix_(self.legs_qvel_idx.FR, self.legs_qvel_idx.FR)],
                                    RL=mass_matrix[np.ix_(self.legs_qvel_idx.RL, self.legs_qvel_idx.RL)],
                                    RR=mass_matrix[np.ix_(self.legs_qvel_idx.RR, self.legs_qvel_idx.RR)])
        return legs_mass_matrix

    @property
    def legs_qfrc_bias(self, ):
        # centrifugal, coriolis, gravity
        legs_qfrc_bias = LegsAttr(FL=self.mjData.qfrc_bias[self.legs_qvel_idx.FL],
                                  FR=self.mjData.qfrc_bias[self.legs_qvel_idx.FR],
                                  RL=self.mjData.qfrc_bias[self.legs_qvel_idx.RL],
                                  RR=self.mjData.qfrc_bias[self.legs_qvel_idx.RR])
        return legs_qfrc_bias

    @property
    def com(self):
        """Calculate the center of mass (CoM) of the entire robot in world frame."""
        total_mass = 0.0
        com = np.zeros(3)
        for i in range(self.mjModel.nbody):
            body_mass = self.mjModel.body_mass[i]
            body_com = self.mjData.subtree_com[i]
            com += body_mass * body_com
            total_mass += body_mass
        com /= total_mass
        return com

    @property
    def base_configuration(self):
        """Robot base configuration (homogenous transformation matrix) in world reference frame."""
        com_pos = self.mjData.qpos[0:3]  # world frame
        quat_wxyz = self.mjData.qpos[3:7]  # world frame (wxyz) mujoco convention
        quat_xyzw = np.roll(quat_wxyz, -1)  # SciPy convention (xyzw)
        X_B = np.eye(4)
        X_B[0:3, 0:3] = Rotation.from_quat(quat_xyzw).as_matrix()
        X_B[0:3, 3] = com_pos
        return X_B

    @property
    def joint_space_state(self):
        """Returns the joint-space state (qpos, qvel) of the robot."""
        return self.mjData.qpos[7:], self.mjData.qvel[6:]

    @property
    def base_pos(self):
        """Returns the base position (3,) in the world reference frame."""
        return self.mjData.qpos[0:3]

    @property
    def base_ori_euler_xyz(self):
        """Returns the base orientation in Euler XYZ angles (roll, pitch, yaw) in the world reference frame."""
        quat_wxyz = self.mjData.qpos[3:7]
        quat_xyzw = np.roll(quat_wxyz, -1)
        return Rotation.from_quat(quat_xyzw).as_euler('xyz')

    @property
    def heading_orientation_SO3(self):
        """Returns a SO(3) matrix that aligns with the robot's base heading orientation and the world z axis."""
        X_B = self.base_configuration
        R_B = X_B[0:3, 0:3]
        euler_xyz = Rotation.from_matrix(R_B).as_euler('xyz')
        # Rotation aligned with the base orientation and the vertical axis
        R_B_horizontal = Rotation.from_euler('xyz', euler_xyz * [0, 0, 1]).as_matrix()
        return R_B_horizontal

    @property
    def torque_ctrl_setpoint(self):
        """Returns the true joint torques (self.mjModel.nu) commanded to the robot actuators.

        Differs from the generalized forces used to evolve the simulation when actuators are non-ideal models.
        """
        return np.array(self.mjData.ctrl)

    @property
    def simulation_dt(self):
        """Returns the simulation dt in seconds."""
        return self.mjModel.opt.timestep

    @property
    def simulation_time(self):
        """Returns the simulation time in seconds."""
        return self.mjData.time
    
    @property
    def robot_model(self):
        """Returns the Robot model."""
        return self.mjModel

    @property
    def sim_data(self):
        """Returns the simulation Data."""
        return self.mjData

    def extract_obs_from_state(self, state_like_array: np.ndarray) -> dict[str, np.ndarray]:
        """Extracts the state observation from a state-like array.

        Parameters
        ----------
        state_like_array : np.ndarray
            The state-like array to extract the state observation from. The shape of
            the array should be (..., self.observation_space.shape).

        Returns
        -------
        state_obs_dict: (dict) A dictionary with the state observation names as keys and the corresponding
            state observation values as values.
        """
        state_obs_shape = self.observation_space.shape
        assert state_like_array.shape[-len(state_obs_shape):] == self.observation_space.shape, \
            f"Invalid state vector shape: {state_like_array.shape}, expected: (...{self.observation_space.shape})"
        state_obs_dict = {}
        # return StateObs(*[state_like_array[..., idx] for idx in self.state_obs_idx.values()])
        return {obs_name: state_like_array[..., idx] for obs_name, idx in self.state_obs_idx.items()}

    def _compute_reward(self):
        # Example reward function (to be defined based on the task)
        # Reward could be based on distance traveled, energy efficiency, etc.
        return 0

    def _get_obs(self):
        """Returns the state observation based on the specified state observation names."""
        obs = []
        for obs_name in self.state_obs_names:
            # Generalized position, velocity, and force (torque) spaces
            if obs_name == 'qpos':
                obs.append(self.mjData.qpos)
            elif obs_name == 'qvel':
                obs.append(self.mjData.qvel)
            elif obs_name == 'tau_ctrl_setpoint':
                obs.append(self.torque_ctrl_setpoint)
            # Joint-space position and velocity spaces
            elif obs_name == 'qpos_js':
                obs.append(self.mjData.qpos[7:])
            elif obs_name == 'qvel_js':
                obs.append(self.mjData.qvel[6:])
            # Base position and velocity configurations (in world frame)
            elif obs_name == 'base_pos':
                obs.append(self.base_pos)
            elif 'base_lin_vel' in obs_name:
                frame = 'world' if not obs_name.endswith('base') else 'base'
                obs.append(self.base_lin_vel(frame))
            elif 'base_lin_acc' in obs_name:
                frame = 'world' if not obs_name.endswith('base') else 'base'
                obs.append(self.base_lin_acc(frame))
            elif obs_name == 'base_ang_vel':
                frame = 'world' if not obs_name.endswith('base') else 'base'
                obs.append(self.base_ang_vel(frame=frame))
            elif obs_name == 'base_ori_euler_xyz':
                obs.append(self.base_ori_euler_xyz)
            elif obs_name == 'base_ori_quat_wxyz':
                obs.append(self.mjData.qpos[3:7])
            elif obs_name == 'base_ori_SO3':
                obs.append(self.base_configuration[0:3, 0:3].flatten())
            # Feet positions and velocities
            elif 'feet_pos' in obs_name:  # feet_pos:frame := feet_pos:world or feet_pos:base
                frame = 'world' if 'base' not in obs_name else 'base'
                obs.append(np.concatenate(self.feet_pos(frame).to_list(order=self.legs_order), axis=0))
            elif 'feet_vel' in obs_name:  # feet_vel:frame := feet_vel:world or feet_vel:base
                frame = 'world' if 'base' not in obs_name else 'base'
                obs.append(np.concatenate(self.feet_vel(frame).to_list(order=self.legs_order), axis=0))
            elif obs_name == 'contact_state':
                contact_state, _ = self.feet_contact_state()
                obs.append(np.array(contact_state.to_list()))
            elif 'contact_forces' in obs_name:
                frame = 'world' if 'base' not in obs_name else 'base'
                _, _, contact_forces = self.feet_contact_state(ground_reaction_forces=True, frame=frame)
                obs.append(np.concatenate(contact_forces.to_list(order=self.legs_order), axis=0))
            else:
                raise ValueError(f"Invalid observation name: {obs_name}, available obs: {self.ALL_OBS}")

        state_obs = np.concatenate(obs, axis=0)
        assert state_obs.shape == self.observation_space.shape, \
            f"Invalid state observation shape: {state_obs.shape}, expected: {self.observation_space.shape}"
        return state_obs

    def _check_for_invalid_contacts(self) -> [bool, dict]:
        """Env termination occurs when a contact is detected on the robot's base."""
        invalid_contacts = {}
        invalid_contact_detected = False
        for contact in self.mjData.contact:
            # Get body IDs from geom IDs
            body1_id = self.mjModel.geom_bodyid[contact.geom1]
            body2_id = self.mjModel.geom_bodyid[contact.geom2]

            if 0 in [body1_id, body2_id]:  # World body ID is 0
                second_id = body2_id if body1_id == 0 else body1_id
                if second_id not in self._feet_body_id.to_list():  # Check if contact occurs with anything but the feet
                    # Get body names from body IDs
                    body1_name = mujoco.mj_id2name(self.mjModel, mujoco.mjtObj.mjOBJ_BODY, body1_id)
                    body2_name = mujoco.mj_id2name(self.mjModel, mujoco.mjtObj.mjOBJ_BODY, body2_id)
                    invalid_contacts[f"{body1_name}:{body1_id}_{body2_name}:{body2_id}"] = contact
                    invalid_contact_detected = True
            else:  # Contact between two bodies of the robot
                pass  # Do nothing for now

        return invalid_contact_detected, invalid_contacts  # No invalid contact detected

    def _get_geom_body_info(self, geom_name: str = None, geom_id: int = None) -> [int, str]:
        """Returns the body ID and name associated with the geometry name or ID."""
        assert geom_name is not None or geom_id is not None, "Please provide either the geometry name or ID."
        if geom_name is not None:
            geom_id = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

        body_id = self.mjModel.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(self.mjModel, mujoco.mjtObj.mjOBJ_BODY, body_id)

        return body_id, body_name

    def _update_camera_target(self, cam, target_point: np.ndarray):
        cam.lookat[:] = target_point  # Update the camera lookat point to the target point
        # Potentially do other fancy stuff.
        pass

    # Function to update the camera position

    def _set_ground_friction(self,
                             tangential_coeff: float = 1.0,  # Default MJ tangential coefficient
                             torsional_coeff: float = 0.005,  # Default MJ torsional coefficient
                             rolling_coeff: float = 0.0  # Default MJ rolling coefficient
                             ):
        """Initialize ground friction coefficients using a specified distribution."""
        pass
        for geom_id in range(self.mjModel.ngeom):
            geom_name = mujoco.mj_id2name(self.mjModel, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name and geom_name.lower() in ['ground', 'floor', 'hfield', "terrain"]:
                self.mjModel.geom_friction[geom_id, :] = [tangential_coeff, torsional_coeff, rolling_coeff]
                # print(f"Setting friction for {geom_name} to: {tangential_coeff, torsional_coeff, rolling_coeff}")
            elif geom_id in self._feet_geom_id:  # Set the same friction coefficients for the feet geometries
                self.mjModel.geom_friction[geom_id, :] = [tangential_coeff, torsional_coeff, rolling_coeff]
            else:
                pass

    def _render_ghost_robots(self, qpos: np.ndarray, alpha: float | np.ndarray = 0.5):
        """ Render ghost robots with the provided qpos configurations.

        :param qpos: (n_robots, nq) or (nq,) array with the joint positions of the ghost robots.
        :param alpha: (float) or (n_robots,) array with the transparency of the ghost robots.
        """
        qp = np.asarray(qpos)
        if qp.ndim == 2:
            assert qp.shape[1] == self.mjModel.nq, f"Invalid qpos shape: (...,{qp.shape[1]}) != {self.mjModel.nq}"
            alpha = [alpha] * qp.shape[0] if isinstance(alpha, float) else alpha
        else:
            qp = qp.reshape(1, -1)
            alpha = [alpha]

        for ghost_robot_idx, (q, a) in enumerate(zip(qp, alpha)):
            # Use forward kinematics to update the geometry positions
            if ghost_robot_idx not in self._ghost_robots_geom:
                self._ghost_robots_geom[ghost_robot_idx] = {}

            self._ghost_mjData.qpos = q
            self._ghost_mjData.qvel *= 0.0
            mujoco.mj_forward(self.mjModel, self._ghost_mjData)
            self._ghost_robots_geom[ghost_robot_idx] |= render_ghost_robot(
                self.viewer,
                self.mjModel,
                self._ghost_mjData,
                alpha=a,
                ghost_geoms=self._ghost_robots_geom.get(ghost_robot_idx, {})
                )

    def _key_callback(self, keycode):
        # print(f"\n\n ********************* Key pressed: {keycode}\n\n\n")
        if keycode == 262:  # arrow right
            self._ref_base_ang_yaw_dot -= np.pi / 6
        elif keycode == 263:  # arrow left
            self._ref_base_ang_yaw_dot += np.pi / 6
        elif keycode == 265:  # arrow up
            self._ref_base_lin_vel_H[0] += 0.25 * self.hip_height  # % of (hip_height / second)
        elif keycode == 264:  # arrow down
            self._ref_base_lin_vel_H[0] -= 0.25 * self.hip_height  # % of (hip_height / second)
        elif keycode == 345:  # ctrl
            self._ref_base_lin_vel_H *= 0.0
            self._ref_base_ang_yaw_dot = 0.0

        self._ref_base_ang_yaw_dot = np.clip(self._ref_base_ang_yaw_dot, -2 * np.pi, 2 * np.pi)
        self._ref_base_lin_vel_H[0] = np.clip(self._ref_base_lin_vel_H[0], -6 * self.hip_height, 6 * self.hip_height)

    def _configure_observation_space(self, state_obs_names: list[str]) -> [spaces.Space, dict[str, slice]]:
        """Configures the observation space for the environment based on the provided state observation names.

        Args:
        ----
        state_obs_names (list[str]): A list of state observation names based on which the observation space is
        configured.

        Returns:
        -------
        gym.Space: The environment state observation space.
        dict: A dictionary mapping each state observation name to its indices in the observation space.
        """
        obs_dim, last_idx = 0, 0

        obs_lim_min, obs_lim_max = [], []
        qpos_lim_min, qpos_lim_max = self.mjModel.jnt_range[:, 0], self.mjModel.jnt_range[:, 1]
        tau_lim_min, tau_lim_max = self.mjModel.actuator_ctrlrange[:, 0], self.mjModel.actuator_ctrlrange[:, 1]

        obs_idx = {k: None for k in state_obs_names}
        for obs_name in state_obs_names:
            # Generalized position, velocity, and force (torque) spaces

            if obs_name == 'qpos':
                obs_dim += self.mjModel.nq
                obs_lim_max.extend([np.inf] * 7 + qpos_lim_max[1:].tolist())  # Ignore the base position
                obs_lim_min.extend([-np.inf] * 7 + qpos_lim_min[1:].tolist())  # Ignore the base position
            elif obs_name == 'qvel':
                obs_dim += self.mjModel.nv
                obs_lim_max.extend([np.inf] * self.mjModel.nv)
                obs_lim_min.extend([-np.inf] * self.mjModel.nv)
            elif obs_name == 'tau_ctrl_setpoint':
                obs_dim += self.mjModel.nu
                obs_lim_max.extend(tau_lim_max)
                obs_lim_min.extend(tau_lim_min)
            # Joint-space position and velocity spaces
            elif obs_name == 'qpos_js':  # Joint space position configuration
                obs_dim += self.mjModel.nq - 7
                obs_lim_max.extend(qpos_lim_max[1:])
                obs_lim_min.extend(qpos_lim_min[1:])
            elif obs_name == 'qvel_js':  # Joint space velocity configuration
                obs_dim += self.mjModel.nv - 6
                obs_lim_max.extend([np.inf] * (self.mjModel.nv - 6))
                obs_lim_min.extend([-np.inf] * (self.mjModel.nv - 6))
            # Base position and velocity configurations (in world frame)
            elif obs_name == 'base_pos':
                if "qpos" in state_obs_names:
                    warnings.warn("base_pos is redundant with additional obs qpos. base_pos = qpos[0:3]")
                obs_dim += 3
                obs_lim_max.extend([np.inf] * 3)
                obs_lim_min.extend([-np.inf] * 3)
            elif 'base_lin_vel' in obs_name:  # base_lin_vel / base_lin_vel:base (base frame)
                if "qvel" in state_obs_names:
                    warnings.warn("base_lin_vel is redundant with additional obs qvel. base_lin_vel = qvel[0:3]")
                obs_dim += 3
                obs_lim_max.extend([np.inf] * 3)
                obs_lim_min.extend([-np.inf] * 3)
            elif 'base_lin_acc' in obs_name:  # base_lin_acc / base_lin_acc:base (base frame)
                obs_dim += 3
                obs_lim_max.extend([np.inf] * 3)
                obs_lim_min.extend([-np.inf] * 3)
            elif obs_name == 'base_ang_vel':
                if "qvel" in state_obs_names:
                    warnings.warn("base_ang_vel is redundant with additional obs qvel. base_ang_vel = qvel[3:6]")
                obs_dim += 3
                obs_lim_max.extend([np.inf] * 3)
                obs_lim_min.extend([-np.inf] * 3)
            elif obs_name == 'base_ori_euler_xyz':
                if "qpos" in state_obs_names:
                    warnings.warn(
                        "base_ori_euler_xyz is redundant with additional obs qpos. base_ori_euler_xyz = qpos[3:6]")
                obs_dim += 3
                obs_lim_max.extend([np.inf] * 3)
                obs_lim_min.extend([-np.inf] * 3)
            elif obs_name == 'base_ori_quat_wxyz':
                if "qpos" in state_obs_names:
                    warnings.warn(
                        "base_ori_quat_wxyz is redundant with additional obs qpos. base_ori_quat_wxyz = qpos[3:7]")
                obs_dim += 4
                obs_lim_max.extend([np.inf] * 4)
                obs_lim_min.extend([-np.inf] * 4)
            elif obs_name == 'base_ori_SO3':
                if "qpos" in state_obs_names:
                    warnings.warn("base_ori_SO3 is redundant with additional obs qpos. base_ori_SO3 = qpos[3:7]")
                obs_dim += 9
                obs_lim_max.extend([np.inf] * 9)
                obs_lim_min.extend([-np.inf] * 9)
            # Feet positions and velocities
            elif 'feet_pos' in obs_name:  # feet_pos:frame := feet_pos:world or feet_pos:base
                obs_dim += 12
                obs_lim_max.extend([np.inf] * 12)
                obs_lim_min.extend([-np.inf] * 12)
            elif 'feet_vel' in obs_name:  # feet_vel:frame := feet_vel:world or feet_vel:base
                obs_dim += 12
                obs_lim_max.extend([np.inf] * 12)
                obs_lim_min.extend([-np.inf] * 12)
            elif obs_name == 'contact_state':
                obs_dim += 4
                obs_lim_max.extend([1] * 4)
                obs_lim_min.extend([0] * 4)
            elif 'contact_forces' in obs_name:
                obs_dim += 12
                obs_lim_max.extend([np.inf] * 12)
                obs_lim_min.extend([-np.inf] * 12)
            else:
                raise ValueError(f"Invalid observation name: {obs_name}, available obs: {self.ALL_OBS}")
            obs_idx[obs_name] = range(last_idx, obs_dim)
            last_idx = obs_dim

            if obs_dim != len(obs_lim_max) or obs_dim != len(obs_lim_min):
                raise ValueError(
                    f"Invalid configuration of observation {obs_name}: \n - obs_dim: {obs_dim} \n"
                    f" - lower_lim_dim: {len(obs_lim_max)} \t - upper_lim_dim: {len(obs_lim_min)}"
                    )

        obs_lim_min = np.array(obs_lim_min)
        obs_lim_max = np.array(obs_lim_max)
        observation_space = spaces.Box(low=obs_lim_min, high=obs_lim_max, shape=(obs_dim,), dtype=np.float32)
        return observation_space, obs_idx

    def _save_hyperparameters(self, constructor_params):
        self._init_args = constructor_params
        [self._init_args.pop(k) for k in ['self', '__class__']]  # Remove 'self' and '__class__

    def _find_feet_model_attrs(self, feet_geom_name):
        _all_geoms = [mujoco.mj_id2name(self.mjModel, i, mujoco.mjtObj.mjOBJ_GEOM) for i in range(self.mjModel.ngeom)]
        for lef_name in ["FR", "FL", "RR", "RL"]:
            foot_geom_id = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_GEOM, feet_geom_name[lef_name])
            assert foot_geom_id != -1, f"Foot GEOM {feet_geom_name[lef_name]} not found in {_all_geoms}."
            self._feet_geom_id[lef_name] = foot_geom_id
            foot_body_id, foot_body_name = self._get_geom_body_info(geom_id=foot_geom_id)
            self._feet_body_id[lef_name] = foot_body_id

    def __str__(self):
        """Returns a description of the environment task configuration."""
        msg = f"robot={self._init_args['robot']} terrain={self._init_args['scene']} task={self.base_vel_command_type}"
        if self.base_vel_command_type == 'human':
            pass
        else:
            msg += (f" lin_vel_range=({self.base_lin_vel_range[0]:.3f}, {self.base_lin_vel_range[1]:.3f})"
                    f" ang_vel_range=({self.base_ang_vel_range[0]:.3f}, {self.base_ang_vel_range[1]:.3f})"
                    f" lat_friction_range=({self.ground_friction_coeff_range[0]:.1e}, "
                    f"{self.ground_friction_coeff_range[1]:.1e})")
        return msg


# Example usage:
if __name__ == '__main__':

    robot_name = "mini_cheetah"
    scene_name = "flat"
    robot_feet_geom_names = dict(FL='FL', FR='FR', RL='RL', RR='RR')
    robot_leg_joints = dict(FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', ],  # TODO: Make configs per robot.
                            FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', ],
                            RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', ],
                            RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', ])

    state_observables_names = tuple(QuadrupedEnv.ALL_OBS)

    env = QuadrupedEnv(robot='mini_cheetah',
                       hip_height=0.25,
                       legs_joint_names=robot_leg_joints,        # Joint names of the legs DoF
                       feet_geom_name=robot_feet_geom_names,     # Geom/Frame id of feet
                       scene=scene_name,
                       ref_base_lin_vel=(0.5, 1.0),              # pass a float for a fixed value
                       ground_friction_coeff=(0.2, 1.5),         # pass a float for a fixed value
                       base_vel_command_type="random",           # "forward", "random", "forward+rotate", "human"
                       state_obs_names=state_observables_names,  # Desired quantities in the 'state'
                       )
    obs = env.reset()
    env.render(tint_robot=True)
    for _ in range(10):
        obs = env.reset()
        for _ in range(20000):
            qpos, qvel = env.mjData.qpos, env.mjData.qvel

            action = env.action_space.sample() * 50  # Sample random action
            state, reward, is_terminated, is_truncated, info = env.step(action=action)

            state_dict = env.extract_obs_from_state(state)
            for state_obs_name in state_observables_names:
                assert state_obs_name in state_dict, f"Missing state observation: {state_obs_name}"
                assert state_dict[state_obs_name] is not None, f"Invalid state observation: {state_obs_name}"

            if is_terminated:
                pass
                # Handle terminal states here. Terminal states are contacts with ground with any geom but feet.

            # The environment enables also to visualize ghost robot configurations for debugging purposes.
            # These ghost/decorative robots are not simulated, rather only displayed in the viewer.
            # These robot's config are given by a qpos array.
            qpos_ghost1, qpos_ghost2 = np.array(qpos), np.array(qpos)
            qpos_ghost1[0] += 1.0
            qpos_ghost2[0] -= 1.0
            env.render(ghost_qpos=(qpos_ghost1, qpos_ghost2), ghost_alpha=(0.1, 0.5))

    env.close()

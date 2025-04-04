# Created by danfoa at 11/02/25
# Description: This script is used to simulate the full model of the robot in mujoco

# Authors:
# Giulio Turrisi, Daniel Ordonez

import pathlib
import time
from os import PathLike
from pprint import pprint

import numpy as np

# PyMPC controller imports
from tqdm import tqdm

# Gym and Simulation related imports
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.data.h5_dataset import H5Reader, H5Writer


def collate_obs(list_of_dicts):
    """Collates a list of dictionaries containing observation names and numpy arrays into a single dictionary of stacked
    numpy arrays.
    """
    if not list_of_dicts:
        raise ValueError('Input list is empty.')

    # Get all keys (assumes all dicts have the same keys)
    keys = list_of_dicts[0].keys()

    # Stack the values per key
    collated = {key: np.stack([d[key] for d in list_of_dicts], axis=0) for key in keys}

    return collated


def record_dataset(
    env: QuadrupedEnv,
    dataset_path: PathLike = None,
    num_episodes=3,
    max_steps=500,
    render=True,
):
    print('Recording dataset with the observations')
    pprint(env.get_hyperparameters()['state_obs_names'])
    # We create a H5 dataset file configured to expect observations from the environment observation and action space.
    h5_manager = H5Writer(
        file_path=dataset_path,
        env=env,
    )

    env.reset(random=False)
    RENDER_FREQ = 30
    if render:
        env.render()  # Pass in the first render call any mujoco.viewer.KeyCallbackType

    for episode_num in range(num_episodes):
        ep_state_history = []
        ep_action_history = []
        ep_time = []
        last_render_time = time.time()
        for _ in tqdm(range(max_steps), desc=f'Recording data. Episode:{episode_num:d}', total=max_steps):
            time.time()

            # Your custom control policy here ---------------------------------------------------------------------
            action = env.action_space.sample()  # Sample random action

            # Evolve the simulation --------------------------------------------------------------------------------
            state, reward, is_terminated, is_truncated, info = env.step(action=action)
            # Store the env state and simulation time stamp
            ep_state_history.append(state)
            ep_action_history.append(action)
            ep_time.append(env.simulation_time)

            # Render only at a certain frequency -----------------------------------------------------------------
            if render and (time.time() - last_render_time > 1.0 / RENDER_FREQ or env.step_num == 1):
                env.render()
                last_render_time = time.time()

            # Reset the environment if the episode is terminated ------------------------------------------------
            if env.step_num >= max_steps or is_terminated:
                break

        # Episode end. Save data to the h5 file
        ep_traj_state_history = collate_obs(ep_state_history)
        ep_traj_state_history['action'] = np.asarray(ep_action_history)
        ep_traj_time = np.asarray(ep_time)[:, np.newaxis]

        # Save the entire state observation trajectory to disk.
        h5_manager.append_trajectory(state_obs_traj=ep_traj_state_history, time=ep_traj_time)

        env.reset()
    env.close()
    return dataset_path


if __name__ == '__main__':
    # if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(0)

    robot_name = 'mini_cheetah'
    scene_name = 'flat'
    robot_feet_geom_names = {'FL': 'FL', 'FR': 'FR', 'RL': 'RL', 'RR': 'RR'}
    robot_leg_joints = {
        'FL': [
            'FL_hip_joint',
            'FL_thigh_joint',
            'FL_calf_joint',
        ],  # TODO: Make configs per robot.
        'FR': [
            'FR_hip_joint',
            'FR_thigh_joint',
            'FR_calf_joint',
        ],
        'RL': [
            'RL_hip_joint',
            'RL_thigh_joint',
            'RL_calf_joint',
        ],
        'RR': [
            'RR_hip_joint',
            'RR_thigh_joint',
            'RR_calf_joint',
        ],
    }

    state_observables_names = tuple(QuadrupedEnv.ALL_OBS)

    env = QuadrupedEnv(
        robot='mini_cheetah',
        hip_height=0.25,
        legs_joint_names=robot_leg_joints,  # Joint names of the legs DoF
        feet_geom_name=robot_feet_geom_names,  # Geom/Frame id of feet
        scene=scene_name,
        ref_base_lin_vel=(0.5, 1.0),  # pass a float for a fixed value
        ground_friction_coeff=(0.2, 1.5),  # pass a float for a fixed value
        base_vel_command_type='random',  # "forward", "random", "forward+rotate", "human"
        state_obs_names=state_observables_names,  # Desired quantities in the 'state'
    )

    n_episodes = 3
    max_steps = 500
    render = False
    # Record multiple episodes of the env
    dataset_path = pathlib.Path(f'data/{robot_name}/proprioceptive_data.h5')

    data_path = record_dataset(
        env=env, dataset_path=dataset_path, render=render, num_episodes=n_episodes, max_steps=max_steps
    )

    # Load the dataset  ==============================================================================================
    dataset = H5Reader(file_path=data_path)
    print(
        f'Loaded a dataset with {dataset.len()} simulated episodes and observations: \n{dataset.env_hparams["state_obs_names"]}'
    )

    # You can access each observation data using a numpy like interface
    obs_names = dataset.env_hparams['state_obs_names']
    for obs_name in obs_names:
        obs_data = dataset.recordings[obs_name]
        print(f'Observation: {obs_name} \n Recorded data shape: {obs_data.shape}')

    # Given the dataset file, we can recreate the environment used to generate the data ==============================
    reproduced_env = QuadrupedEnv(**dataset.env_hparams)
    # We can also list the observations in the dataset

    # And since we use the same names as QuadrupedEnv, we can get the group representations for free =================
    from gym_quadruped.utils.quadruped_utils import configure_observation_space_representations

    obs_reps = configure_observation_space_representations(robot_name=dataset.env_hparams['robot'], obs_names=obs_names)

    for obs_name in obs_names:
        obs_data = dataset.recordings[obs_name]
        print(f'{obs_name} \n \tRecorded data shape: {obs_data.shape}')
        print(f'\tGroup representation acting on this observation: \n \t{obs_reps[obs_name]}')

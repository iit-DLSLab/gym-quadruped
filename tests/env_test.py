# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 02/04/25
import numpy as np
import pytest

from gym_quadruped.quadruped_env import QuadrupedEnv


@pytest.mark.parametrize('robot_name', ['mini_cheetah', 'b2', 'go1', 'go2', 'hyqreal'])
@pytest.mark.parametrize('terrain_type', ['flat', 'perlin'])
def test_robot_env(robot_name, terrain_type):  # noqa: D103
    # TODO: Need to make the config for each robot into yaml files.
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
        robot=robot_name,
        hip_height=0.25,
        legs_joint_names=robot_leg_joints,  # Joint names of the legs DoF
        feet_geom_name=robot_feet_geom_names,  # Geom/Frame id of feet
        scene=terrain_type,
        ref_base_lin_vel=(0.5, 1.0),  # pass a float for a fixed value
        ground_friction_coeff=(0.2, 1.5),  # pass a float for a fixed value
        base_vel_command_type='forward+rotate',  # "forward", "random", "forward+rotate", "human"
        state_obs_names=state_observables_names,  # Desired quantities in the 'state'
    )
    # Test reset

    state = env.reset()
    qpos, qvel = state['qpos'], state['qvel']
    state = env.reset(random=True)
    state = env.reset(qpos=qpos, qvel=qvel)

    # Check all desired observables are present in the state.
    for obs_name in state_observables_names:
        try:
            obs_val = state[obs_name]
            obs_shape = np.asarray(obs_val).shape
            assert obs_shape == env.observation_space[obs_name].shape
        except KeyError as e:
            raise AssertionError(f'Observable {obs_name} not found in the state for the robot {robot_name}') from e
        except AssertionError as e:
            raise AssertionError(
                f'Observable {obs_name} has incorrect shape, expected {env.action_space[obs_name].shape} got '
                f'{np.asarray(obs_val).shape}'
            ) from e

    # Test simulation step.
    for _ in range(10):
        action = env.action_space.sample() * 50  # Sample random action
        state, reward, is_terminated, is_truncated, info = env.step(action=action)

    env.close()

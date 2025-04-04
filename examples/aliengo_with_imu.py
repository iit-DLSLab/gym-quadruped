"""Copyright (c) 2025 Hilton-Santana <https://my.github.com/Hilton-Santana>

Created Date: Saturday, January 18th 2025, 4:08:51 pm
Author: Hilton-Santana hiltonmarquess@gmail.com

Description: A simple example of using the Aliengo environment with an IMU sensor.
HISTORY:
Date      	By	Comments
----------	---	----------------------------------------------------------
"""

from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.sensors.imu import IMU

robot_name = 'aliengo'  # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = 'flat'

robot_feet_geom_names = {'FL': 'FL', 'FR': 'FR', 'RL': 'RL', 'RR': 'RR'}
robot_leg_joints = {
    'FL': [
        'FL_hip_joint',
        'FL_thigh_joint',
        'FL_calf_joint',
    ],
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


imu_kwargs = {
    'accel_name': 'Body_Acc',
    'gyro_name': 'Body_Gyro',
    'imu_site_name': 'imu',
    'accel_noise': 0.01,
    'gyro_noise': 0.01,
    'accel_bias_rate': 0.01,
    'gyro_bias_rate': 0.01,
}

state_observables_names = ('qpos', 'qvel') + tuple(IMU.ALL_OBS)  # return all available state observables
env = QuadrupedEnv(
    robot=robot_name,
    hip_height=0.25,
    legs_joint_names=robot_leg_joints,  # Joint names of the legs DoF
    feet_geom_name=robot_feet_geom_names,  # Geom/Frame id of feet
    scene=scene_name,
    ref_base_lin_vel=0.5,  # Constant magnitude of reference base linear velocity [m/s]
    base_vel_command_type='forward',  # "forward", "random", "forward+rotate", "human"
    state_obs_names=state_observables_names,  # Desired quantities in the 'state'
    sensors=(IMU,),  # Add IMU sensor to the environment
    sensors_kwargs=(imu_kwargs,),  # Pass the IMU sensor kwargs
)

obs = env.reset()
env.render()

dt = 0.1
t0 = 0.0
while True:
    # Sensor gets updated only at the evolution of the simulation
    action = env.action_space.sample() * 50  # Sample random action
    state, reward, is_terminated, is_truncated, info = env.step(action=action)
    env.render()
    imu_acc, imu_acc_noise, imu_acc_bias = (
        state['imu_acc'],
        state['imu_acc_noise'],
        state['imu_acc_bias'],
    )
    imu_gyro, imu_gyro_noise, imu_gyro_bias = (
        state['imu_gyro'],
        state['imu_gyro_noise'],
        state['imu_gyro_bias'],
    )
    print(f't: {t0:.2f}, \n\taccel: {imu_acc}, \n\tgyro: {imu_gyro}')

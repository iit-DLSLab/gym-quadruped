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
from gym_quadruped.robot_cfgs import RobotConfig, get_robot_config

robot_name = 'aliengo'  # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = 'slippery' # "flat", "stairs", "ramp", "perlin", "random_boxes", "random_pyramids"

robot_cfg: RobotConfig = get_robot_config(robot_name=robot_name)
robot_leg_joints = robot_cfg.leg_joints


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

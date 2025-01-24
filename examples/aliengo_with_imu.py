'''
Copyright (c) 2025 Hilton-Santana <https://my.github.com/Hilton-Marques>

Created Date: Saturday, January 18th 2025, 4:08:51 pm
Author: Hilton-Santana hiltonmarquess@gmail.com

Description: A simple example of using the Aliengo environment with an IMU sensor.
HISTORY:
Date      	By	Comments
----------	---	----------------------------------------------------------
'''

from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.sensors.imu import IMU

robot_name = "aliengo"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = "flat"

robot_feet_geom_names = dict(FL='FL', FR='FR', RL='RL', RR='RR')
robot_leg_joints = dict(FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', ], 
                        FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', ],
                        RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', ],
                        RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', ])

state_observables_names = tuple(QuadrupedEnv.ALL_OBS)  # return all available state observables

env = QuadrupedEnv(robot=robot_name,
                   hip_height=0.25,
                   legs_joint_names=robot_leg_joints,  # Joint names of the legs DoF
                   feet_geom_name=robot_feet_geom_names,  # Geom/Frame id of feet
                   scene=scene_name,
                   ref_base_lin_vel=0.5, # Constant magnitude of reference base linear velocity [m/s]
                   base_vel_command_type="forward",  # "forward", "random", "forward+rotate", "human"
                   state_obs_names=state_observables_names,  # Desired quantities in the 'state'
                   )

obs = env.reset()
env.render()

imu = IMU(mj_model=env.mjModel,
          mj_data=env.mjData,
          accel_name="Body_Acc",
          gyro_name="Body_Gyro",
          imu_site_name="imu")

imu.prepare2show()

dt = 0.1
t0 = 0.0
while True:
  imu_accel, accel_noise, accel_bias  = imu.get_accel
  imu_gyro, gyro_noise, gyro_bias  = imu.get_gyro
  X = imu.get_imu_frame

  imu.show(t0,
           accel_noise,
           gyro_noise,
           accel_bias,
           gyro_bias)

  t0 += dt
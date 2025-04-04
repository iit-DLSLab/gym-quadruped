# Mujoco Gym Environments for quadupedal legged locomotion

# Install Instructions

```bash
git clone https://github.com/iit-DLSLab/gym-quadruped.git
cd gym-quadruped
pip install -e .   # This will install the package in editable mode
```

# Usage Instructions

```python
from gym_quadruped.quadruped_env import QuadrupedEnv

robot_name = "mini_cheetah"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = "flat"
robot_feet_geom_names = dict(FL='FL', FR='FR', RL='RL', RR='RR')
robot_leg_joints = dict(FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', ],  # TODO: Make configs per robot.
                        FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', ],
                        RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', ],
                        RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', ])

state_observables_names = tuple(QuadrupedEnv.ALL_OBS)  # return all available state observables

env = QuadrupedEnv(robot='mini_cheetah',
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
for _ in range(10000):
    action = env.action_space.sample() * 50  # Sample random action
    state, reward, is_terminated, is_truncated, info = env.step(action=action)

    if is_terminated:
        pass
        # Do some stuff
    env.render()
env.close()
```

# Package-Name suggestions accepted

The end idea of this package is to be able to have a Mujoco Gym Environment to test model-based and learnign controllers
for locomotion in controlled environments with shared metrics (e.g., energy efficiency, robustness, etc.)

The core principles should be:

- We (will) use `robot_descriptions.py` to load the robot URDF/SDF files and create the Mujoco model.
- We programatically generate terrains and obstacles to test the robot in different scenarios.
- We offer a suit of metrics to evaluate and compare the performance of different controllers with a simple API.
- THIS IS A LIGHTWEIGHT PACKAGE. all custom stuff should be defined elsewhere.
- We will publish datasets of expert model-based/learning controlled trajectories using LeRobot (project structure
  inherited from them) and Huggingface API.

At some point we should generate mujoco jax versions of the environments for simulation and data-collection in GPU.

```
Death to C++

Att: Giulio Turrisi
```
# Mujoco Gym Environment for quadupedal legged locomotion

[![PyPI version](https://img.shields.io/pypi/v/gym-quadruped.svg)](https://pypi.org/project/gym-quadruped/) [![Python Version](https://img.shields.io/badge/python-3.10%20--%203.12-blue)](https://github.com/Danfoa/MorphoSymm/actions/workflows/tests.yaml)

# Install Instructions

```bash
  pip install gym-quadruped
  # or install locally
  cd <gym-quadruped root dir> 
  pip install -e . 
```

# Usage Instructions

```python
from gym_quadruped.quadruped_env import QuadrupedEnv

robot_name = "mini_cheetah"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = "flat"  # perlin | random_boxes
state_observables_names = tuple(QuadrupedEnv.ALL_OBS)  # return all available state observables

env = QuadrupedEnv(robot='mini_cheetah',
                   scene=scene_name,
                   base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
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

See also `examples` directory.
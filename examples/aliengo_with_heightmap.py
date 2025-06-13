from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.sensors.heightmap import HeightMap
from gym_quadruped.utils.mujoco.visual import render_sphere
from gym_quadruped.robot_cfgs import RobotConfig, get_robot_config

robot_name = 'aliengo'  # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = 'stairs' # "flat", "stairs", "ramp", "perlin", "random_boxes", "random_pyramids"

robot_cfg: RobotConfig = get_robot_config(robot_name=robot_name)
robot_leg_joints = robot_cfg.leg_joints

state_observables_names = tuple(QuadrupedEnv.ALL_OBS)  # return all available state observables


env = QuadrupedEnv(
    robot=robot_name,
    scene=scene_name,
    ref_base_lin_vel=0.5,  # Constant magnitude of reference base linear velocity [m/s]
    base_vel_command_type='forward',  # "forward", "random", "forward+rotate", "human"
    state_obs_names=state_observables_names,  # Desired quantities in the 'state'
)
obs = env.reset()

# adding the heightmap
heightmap = HeightMap(n=5, dist_x=0.1, dist_y=0.1, mj_model=env.mjModel, mj_data=env.mjData)
env.render()


N_STEPS_PER_EPISODE = 1000
while True:
    # this fixes the base floating mid air at point 1,1,0.5
    env.mjData.qpos[0] = -1
    env.mjData.qpos[1] = 1
    env.mjData.qpos[2] = 0.5

    sim_time = env.simulation_time
    action = env.action_space.sample() * 0  # Sample random action

    state, reward, is_terminated, is_truncated, info = env.step(action=action)

    data = heightmap.update_height_map(env.mjData.qpos[0:3], yaw=env.base_ori_euler_xyz[2])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            heightmap.geom_ids[i, j] = render_sphere(
                viewer=env.viewer,
                position=([data[i][j][0][0], data[i][j][0][1], data[i][j][0][2]]),
                diameter=0.01,
                color=[0, 1, 0, 0.5],
                geom_id=heightmap.geom_ids[i, j],
            )

    if env.step_num > N_STEPS_PER_EPISODE or is_terminated or is_truncated:
        if is_terminated:
            print('Environment terminated')
        else:
            print(f'reset {env.reset_env_counter}')
            env.reset(random=False)
    env.render()
env.close()

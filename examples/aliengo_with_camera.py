import cv2

from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.sensors.rgbd_camera import Camera

robot_name = "aliengo"  # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = "flat"
robot_feet_geom_names = dict(FL="FL", FR="FR", RL="RL", RR="RR")
robot_leg_joints = dict(
    FL=[
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
    ],  # TODO: Make configs per robot.
    FR=[
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
    ],
    RL=[
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ],
    RR=[
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ],
)

state_observables_names = tuple(QuadrupedEnv.ALL_OBS)  # return all available state observables


env = QuadrupedEnv(
    robot=robot_name,
    hip_height=0.25,
    legs_joint_names=robot_leg_joints,  # Joint names of the legs DoF
    feet_geom_name=robot_feet_geom_names,  # Geom/Frame id of feet
    scene=scene_name,
    ref_base_lin_vel=0.5,  # Constant magnitude of reference base linear velocity [m/s]
    base_vel_command_type="forward",  # "forward", "random", "forward+rotate", "human"
    state_obs_names=state_observables_names,  # Desired quantities in the 'state'
)
obs = env.reset()

cam = Camera(
    width=640,
    height=480,
    fps=30,
    model=env.robot_model,
    data=env.sim_data,
    cam_name="robotcam",  # camera must be inserted on the .xml file of the robot in order to work
    save_dir="data_",
)

env.render()
while True:
    sim_time = env.simulation_time
    action = env.action_space.sample() * 0  # Sample random action

    state, reward, is_terminated, is_truncated, info = env.step(action=action)

    if sim_time - cam.last_sim_time >= cam.interval:  # Get camera information based on each camera fps
        # cam.save(depth=True)
        cv2.imshow("image", cam.image)
        cv2.imshow("depth", cam.depth_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        cam.last_sim_time = env.simulation_time

    if is_terminated:
        pass
        # Do some stuff
    env.render()
env.close()

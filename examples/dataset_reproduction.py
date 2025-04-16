# Created by danfoa at 11/02/25

import pathlib
import time
from itertools import chain

import numpy as np
from escnn.group import Representation
from huggingface_hub import hf_hub_download, list_repo_files
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader

from gym_quadruped.data.proprioceptive_datasets import ProprioceptiveDataset
from gym_quadruped.quadruped_env import QuadrupedEnv

# PyMPC controller imports
# Gym and Simulation related imports
from gym_quadruped.utils.data.h5py import H5Reader
from gym_quadruped.utils.mujoco.visual import render_sphere, render_vector
from gym_quadruped.utils.quadruped_utils import LegsAttr, configure_observation_space_representations

import pathlib
import huggingface_hub as hf


def augment_sensor_data(obs_data: dict[str, np.ndarray], obs_reps: dict[str, Representation], fix_base: bool = False):
    assert len(obs_data) > 0
    if fix_base:
        base_pos_t = np.array(obs_data['base_pos'])
        obs_data['base_pos'][..., :2] = 1
        obs_data['qpos'][..., :2] = 1
        obs_data['feet_pos'][..., :2] = obs_data['feet_pos'][..., :2] - base_pos_t[..., :2] + 1
        obs_data['feet_pos'][..., 3:5] = obs_data['feet_pos'][..., 3:5] - base_pos_t[..., :2] + 1
        obs_data['feet_pos'][..., 6:8] = obs_data['feet_pos'][..., 6:8] - base_pos_t[..., :2] + 1
        obs_data['feet_pos'][..., 9:11] = obs_data['feet_pos'][..., 9:11] - base_pos_t[..., :2] + 1
        # obs_data['qpos'][..., 3:7] = np.asarray([0, 1, 1, 1]) / np.linalg.norm(np.asarray([0, 1, 0, 1]))

    G = list(obs_reps.values())[0].group
    G_obs_data = {obs_name: {G.identity: data} for obs_name, data in obs_data.items()}  # e: trivial element
    for g in G.elements[1:]:
        for obs_name, data in obs_data.items():
            if obs_name not in obs_reps or obs_reps[obs_name] is None:
                G_obs_data[obs_name][g] = None
                continue
            real_obs = np.array(G_obs_data[obs_name][G.identity])
            G_obs_data[obs_name][g] = np.einsum('ij,...j->...i', obs_reps[obs_name](g), real_obs)

    # Deal with quaternions.
    rep_R3 = G.representations['R3']
    for g in G.elements[1:]:
        base_ori_quat_xyzw = np.array(G_obs_data['qpos'][G.identity][..., 3:7])
        base_SO3 = Rotation.from_quat(base_ori_quat_xyzw, scalar_first=False).as_matrix()
        g_base_SO3 = rep_R3(g) @ base_SO3 @ rep_R3(g).T
        g_base_ori_quat_xyzw = Rotation.from_matrix(g_base_SO3).as_quat(scalar_first=False)
        qpos = np.concatenate((G_obs_data['base_pos'][g], g_base_ori_quat_xyzw, G_obs_data['qpos_js'][g]), axis=-1)
        G_obs_data['qpos'][g] = qpos

    return G_obs_data


def reproduce_dataset(dataset, fix_base: bool = False):
    np.printoptions(precision=3)

    rec_env = QuadrupedEnv(**dataset.env_hparams)
    rec_env.render(tint_robot=True)

    joint_space_order = list(
        chain.from_iterable([rec_env.robot_cfg.leg_joints[leg_name] for leg_name in rec_env.legs_order])
    )

    obs_reps = configure_observation_space_representations(
        robot_name=rec_env.robot_name.replace('real', ''),
        obs_names=rec_env.state_obs_names,
        joint_space_order=joint_space_order,
    )
    obs_reps['qpos_js'] = obs_reps['qvel_js']  # Hack for mujoco.
    G = list(obs_reps.values())[0].group
    rep_R3 = G.representations['R3']
    rep_QJ = G.representations['Q_js']
    for g in G.elements:
        print(f'{g}: {rep_QJ(g)}')

    geom_ids = {}
    feet_colors = {'FL': [1, 0, 0, 1], 'FR': [0, 1, 0, 1], 'RL': [0, 0, 1, 1], 'RR': [1, 1, 0, 1]}

    n_trajs = dataset.n_trajectories
    for traj_id in range(n_trajs):
        obs_t = {obs_name: dataset.recordings[obs_name][traj_id] for obs_name in rec_env.state_obs_names}

        G_obs_t = augment_sensor_data(obs_t, obs_reps, fix_base=fix_base)

        t_range = np.array(dataset.recordings['time'][traj_id]).squeeze()  # (time, 1)
        rec_env.reset(qpos=G_obs_t['qpos'][G.identity][0], qvel=G_obs_t['qvel'][G.identity][0])

        pc_time0 = time.time()
        frame = 0
        while frame < len(t_range):
            if time.time() - pc_time0 < t_range[frame]:
                continue
            rec_env.reset(qpos=G_obs_t['qpos'][G.identity][frame], qvel=G_obs_t['qvel'][G.identity][frame])

            for g in G.elements:
                g_qpos, g_qvel, g_grf, g_feet_pos = (
                    G_obs_t['qpos'][g],
                    G_obs_t['qvel'][g],
                    G_obs_t['contact_forces'][g],
                    G_obs_t['feet_pos'][g],
                )
                g_feet_vel = G_obs_t['feet_vel'][g]
                feet_pos = LegsAttr(
                    **{leg_name: g_feet_pos[..., i * 3 : i * 3 + 3] for i, leg_name in enumerate(rec_env.legs_order)}
                )
                feet_GRF = LegsAttr(
                    **{leg_name: g_grf[..., i * 3 : i * 3 + 3] for i, leg_name in enumerate(rec_env.legs_order)}
                )
                feet_vel = LegsAttr(
                    **{leg_name: g_feet_vel[..., i * 3 : i * 3 + 3] for i, leg_name in enumerate(rec_env.legs_order)}
                )

                for leg_id, leg_name in enumerate(rec_env.legs_order):
                    geom_ids[f'{g}-grf-{leg_name}'] = render_vector(
                        rec_env.viewer,
                        vector=feet_GRF[leg_name][frame],
                        pos=feet_pos[leg_name][frame],
                        scale=np.linalg.norm(feet_GRF[leg_name][frame]) * 0.005,
                        color=np.array([0, 1, 0, 0.5]),
                        geom_id=geom_ids.get(f'{g}-grf-{leg_name}', -1),
                    )
                    geom_ids[f'{g}-feet_vec_{leg_name}'] = render_vector(
                        rec_env.viewer,
                        vector=feet_vel[leg_name][frame],
                        pos=feet_pos[leg_name][frame],
                        scale=np.linalg.norm(feet_vel[leg_name][frame]) * 0.5,
                        color=np.array([0.9, 0.1, 0.1, 0.5]),
                        geom_id=geom_ids.get(f'{g}-feet_vec_{leg_name}', -1),
                    )
            while rec_env.is_paused:
                time.sleep(0.01)
            rec_env.render(ghost_qpos=[G_obs_t['qpos'][g][frame] for g in G.elements[1:]], ghost_alpha=0.4)
            frame += 1
    rec_env.close()


if __name__ == '__main__':
    # Download the dataset if not present
    datasets_path = pathlib.Path(__file__).parent.parent / 'data'
    test_file = 'data/aliengo/perlin/lin_vel=(0.0, 0.0) ang_vel=(-0.7, 0.7) friction=(1.0, 1.0)/ep=50_steps=2499.h5'
    # if not data_path.exists():
    datasets_path.mkdir(parents=True, exist_ok=True)
    data_path = hf_hub_download(
        repo_id='DLS-IIT/quadruped_locomotion',
        repo_type='dataset',
        filename=test_file,
        local_dir=str(datasets_path),
    )

    dataset = H5Reader(data_path)

    # You can reproduce the dataset in the visualizer TODO: Add realtime replay
    # fix_base is used to properly visualize the data-augmentation.
    fix_base = True
    init_params = dataset.env_hparams
    if fix_base:
        init_params['scene'] = 'flat'
    reproduce_dataset(dataset, fix_base=fix_base)

    # You can also load data to a Torch dataset and use it for your learning projects
    torch_dataset = ProprioceptiveDataset(
        data_file=pathlib.Path(data_path),
        x_obs_names=['base_pos', 'base_ori_SO3', 'qpos_js', 'qvel_js', 'feet_vel:base', 'work', 'kinetic_energy'],
        y_obs_names=['contact_forces:base'],
        x_frames=10,  # Number of timeframes used for your input vector (n_frames, x_obs_dim)
        y_frames=1,  # Number of timeframes used for your output vector (n_frames, y_obs_dim)
        mode='static',
    )

    data_loader = DataLoader(torch_dataset, batch_size=100, shuffle=False)

    for x, y in data_loader:
        for x_obs_name, x_obs in x.items():
            print(f'X - {x_obs_name}: {x_obs.shape}')
        for y_obs_name, y_obs in y.items():
            print(f'Y -{y_obs_name}: {y_obs.shape}')
        break

# Created by danfoa at 11/02/25

import pathlib
import time

import numpy as np
from escnn.group import Representation
from scipy.spatial.transform import Rotation

from gym_quadruped.quadruped_env import QuadrupedEnv

# PyMPC controller imports
# Gym and Simulation related imports
from gym_quadruped.utils.data.h5py import H5Reader
from gym_quadruped.utils.mujoco.visual import render_sphere, render_vector
from gym_quadruped.utils.quadruped_utils import LegsAttr, configure_observation_space_representations


def augment_sensor_data(obs_data: dict[str, np.ndarray], obs_reps: dict[str, Representation]):
    assert len(obs_data) > 0
    G = list(obs_reps.values())[0].group
    G_obs_data = {obs_name: {G.identity: data} for obs_name, data in obs_data.items()}  # e: trivial element
    for g in G.elements[1:]:
        for obs_name, data in obs_data.items():
            if obs_name not in obs_reps or obs_reps[obs_name] is None:
                G_obs_data[obs_name][g] = None
                continue

            G_obs_data[obs_name][g] = np.einsum('ij,...j->...i', obs_reps[obs_name](g), data)

    # Deal with quaternions.
    rep_R3 = G.representations['R3']
    for g in G.elements[1:]:
        base_ori_quat_xyzw = G_obs_data['qpos'][G.identity][..., 3:7]
        base_SO3 = Rotation.from_quat(base_ori_quat_xyzw, scalar_first=False).as_matrix()
        g_base_SO3 = np.einsum('ij,...jl,lm->...im', rep_R3(g), base_SO3, rep_R3(~g))
        g_base_ori_quat_xyzw = Rotation.from_matrix(g_base_SO3).as_quat(scalar_first=False)
        qpos = np.concatenate((G_obs_data['base_pos'][g], g_base_ori_quat_xyzw, G_obs_data['qpos_js'][g]), axis=-1)
        G_obs_data['qpos'][g] = qpos

    return G_obs_data


def reproduce_dataset(dataset):
    rec_env = QuadrupedEnv(**dataset.env_hparams)
    obs_reps = configure_observation_space_representations(
        robot_name=rec_env.robot_name, obs_names=rec_env.state_obs_names
    )
    obs_reps['qpos_js'] = obs_reps['qvel_js']  # Hack for mujoco.
    G = list(obs_reps.values())[0].group

    geom_ids = {}
    feet_colors = {'FL': [1, 0, 0, 1], 'FR': [0, 1, 0, 1], 'RL': [0, 0, 1, 1], 'RR': [1, 1, 0, 1]}

    n_trajs = dataset.n_trajectories
    for traj_id in range(n_trajs):
        obs_t = {obs_name: dataset.recordings[obs_name][traj_id] for obs_name in rec_env.state_obs_names}
        G_obs_t = augment_sensor_data(obs_t, obs_reps)

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
                # g_feet_vel_B = G_obs_t['feet_vel:base'][g]
                feet_pos = LegsAttr(
                    **{leg_name: g_feet_pos[..., i * 3 : i * 3 + 3] for i, leg_name in enumerate(rec_env.legs_order)}
                )
                feet_GRF = LegsAttr(
                    **{leg_name: g_grf[..., i * 3 : i * 3 + 3] for i, leg_name in enumerate(rec_env.legs_order)}
                )
                feet_vel = LegsAttr(
                    **{leg_name: g_feet_vel[..., i * 3 : i * 3 + 3] for i, leg_name in enumerate(rec_env.legs_order)}
                )
                # feet_vel_B = LegsAttr(
                # 	**{leg_name: g_feet_vel_B[..., i * 3 : i * 3 + 3] for i, leg_name in enumerate(rec_env.legs_order)}
                # )

                for leg_id, leg_name in enumerate(rec_env.legs_order):
                    geom_ids[f'{g}-feet_{leg_name}'] = render_sphere(
                        rec_env.viewer,
                        position=feet_pos[leg_name][frame],
                        diameter=0.05,
                        color=feet_colors[leg_name],
                        geom_id=geom_ids.get(f'{g}-feet_{leg_name}', -1),
                    )
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

            rec_env.render(ghost_qpos=[G_obs_t['qpos'][g][frame] for g in G.elements[1:]], ghost_alpha=0.5)
            frame += 1
    rec_env.close()


if __name__ == '__main__':
    path = pathlib.Path(
        '/home/danfoa/Projects/Quadruped-PyMPC/datasets/forward+rotate/go1/terrain=perlin/lin_vel=(2.0, 0.0) '
        'ang_vel=(-0.5, 0.5) friction=(0.9, 1.0)/ep=25_steps=1249.h5'
    )
    dataset = H5Reader(path)

    reproduce_dataset(dataset)

"""Regression coverage for fixed foot bodies and foot kinematics."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import pytest

from gym_quadruped.quadruped_env import QuadrupedEnv

ROBOTS = ['a2', 'aliengo', 'b2', 'go1', 'go2', 'hyqreal1', 'hyqreal2', 'mini_cheetah', 'pegasus', 'spot']


@pytest.mark.parametrize('robot', ROBOTS)
def test_foot_kinematics_and_dynamics(robot):
    """Check body ownership, Jacobians by finite differences, and dynamic observables."""
    env = QuadrupedEnv(robot=robot, scene='flat')
    try:
        env.reset()
        model, data = env.mjModel, env.mjData
        data.qvel[:] = np.linspace(-0.3, 0.4, model.nv)
        mujoco.mj_forward(model, data)
        jac, _ = env.feet_jacobians(return_rot_jac=True)
        jac_dot, rot_dot = env.feet_jacobians_dot(return_rot_jac=True)
        eps = 1e-4
        qpos = data.qpos.copy()
        mujoco.mj_integratePos(model, data.qpos, data.qvel, eps)
        mujoco.mj_forward(model, data)
        next_positions = env.feet_pos()
        next_jac, next_rot = env.feet_jacobians(return_rot_jac=True)
        data.qpos[:] = qpos
        mujoco.mj_integratePos(model, data.qpos, data.qvel, -eps)
        mujoco.mj_forward(model, data)
        previous_positions = env.feet_pos()
        previous_jac, previous_rot = env.feet_jacobians(return_rot_jac=True)
        for leg in env.legs_order:
            body_id = env._feet_body_id[leg]
            assert model.body(body_id).name == f'{leg}_foot'
            assert model.body_jntnum[body_id] == 0
            np.testing.assert_allclose(data.xpos[body_id], data.geom_xpos[env._feet_geom_id[leg]], atol=1e-12)
            np.testing.assert_allclose(
                (next_positions[leg] - previous_positions[leg]) / (2 * eps), jac[leg] @ data.qvel, atol=1e-6
            )
            np.testing.assert_allclose(
                (next_jac[leg] - previous_jac[leg]) / (2 * eps), jac_dot[leg], atol=1e-6
            )
            np.testing.assert_allclose(
                (next_rot[leg] - previous_rot[leg]) / (2 * eps), rot_dot[leg], atol=1e-6
            )
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(env.com, data.subtree_com[1], atol=1e-12)
        mujoco.mj_energyVel(model, data)
        np.testing.assert_allclose(env.kinetic_energy, data.energy[1], rtol=1e-10)
        matrix = np.empty((model.nv, model.nv))
        mujoco.mj_fullM(model, data, matrix)
        np.testing.assert_allclose(env.work, data.qvel @ matrix @ data.qacc)
    finally:
        env.close()


@pytest.mark.parametrize('reverse', [False, True])
def test_foot_contacts_and_force_direction(reverse):
    """Distinguish foot and calf contacts and account for both geom orderings."""
    env = QuadrupedEnv(robot='go1', scene='flat')
    try:
        model = env.mjModel
        foot = env._feet_geom_id.FL
        calf_body = model.body('FL_calf').id
        calf = next(i for i in range(model.ngeom) if model.geom_bodyid[i] == calf_body)
        ground = next(i for i in range(model.ngeom) if model.geom_bodyid[i] == 0)

        def contact(geom):
            return SimpleNamespace(
                geom1=geom if reverse else ground,
                geom2=ground if reverse else geom,
                frame=np.eye(3).ravel(),
            )

        def contact_force(model, data, id, result):
            result[:] = 0
            result[0] = -10 if reverse else 10

        env.mjData = SimpleNamespace(contact=[contact(foot)])
        with patch('gym_quadruped.quadruped_env.mujoco.mj_contactForce', side_effect=contact_force):
            state, _, forces = env.feet_contact_state(ground_reaction_forces=True)
        assert state.FL
        np.testing.assert_allclose(forces.FL, [10, 0, 0])
        assert not env._check_for_invalid_contacts()[0]
        env.mjData.contact = [contact(calf)]
        assert not any(env.feet_contact_state()[0].to_list())
        assert not env._check_for_invalid_contacts()[0]  # Leg contacts do not terminate the episode
        base = next(i for i in range(model.ngeom) if model.geom_bodyid[i] == model.jnt_bodyid[0])
        env.mjData.contact = [contact(base)]
        assert env._check_for_invalid_contacts()[0]
    finally:
        env.close()


@pytest.mark.parametrize(
    'filename',
    ['hyqreal2/hyqreal2_nohpu.xml', 'hyqreal2/hyqreal2_nohpu_nobattery.xml', 'spot/spot_arm.xml'],
)
def test_variant_foot_bodies(filename):
    """Compile model variants and verify their fixed foot bodies."""
    path = Path(__file__).resolve().parents[1] / 'gym_quadruped' / 'robot_model' / filename
    model = mujoco.MjModel.from_xml_path(str(path))
    legs = ['FL', 'FR', 'HL', 'HR'] if 'spot_arm' in filename else ['FL', 'FR', 'RL', 'RR']
    for leg in legs:
        body = model.body(f'{leg}_foot')
        assert model.geom_bodyid[model.geom(leg).id] == body.id
        assert model.body_jntnum[body.id] == 0

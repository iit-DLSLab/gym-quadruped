from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import numpy as np


@dataclass
class RobotConfig:
    """Base class of a quadruped robot configuration."""

    mjcf_filename: str
    hip_height: float  # height of the hip joint in normal stand pose
    qpos0_js: Optional[Iterable] = None  # Zero position of the joint-space configuration
    # Contact points / feet geometries ___________________________________________
    feet_geom_names: dict[str, str] = field(default_factory=lambda: {'FL': 'FL', 'FR': 'FR', 'RL': 'RL', 'RR': 'RR'})
    # Joint names per leg.
    leg_joints: dict[str, list[str]] = field(
        default_factory=lambda: {
            'FL': ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint'],
            'FR': ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint'],
            'RL': ['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'],
            'RR': ['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'],
        }
    )
    # IMU related parameters _______________________________________________________
    accel_name: Optional[str] = None  # accelerometer name in the xml file
    gyro_name: Optional[str] = None  # gyroscope name in the xml file
    imu_site_name: Optional[str] = None  # imu site name in the xml file


def get_robot_config(robot_name: str) -> RobotConfig:
    """Get the robot configuration based on the robot name."""
    name = robot_name.lower()

    if 'mini_cheetah' in name:
        cfg = RobotConfig(
            mjcf_filename='mini_cheetah/mini_cheetah.xml',
            hip_height=0.225,
            qpos0_js=[0, -np.pi / 2, 0] * 2 + [0, np.pi / 2, 0] * 2,
        )
    elif name == 'go1':
        cfg = RobotConfig(mjcf_filename='go1/go1.xml', hip_height=0.3)
    elif name == 'go2':
        cfg = RobotConfig(mjcf_filename='go2/go2.xml', hip_height=0.28)
    elif name == 'aliengo':
        cfg = RobotConfig(mjcf_filename='aliengo/aliengo.xml', hip_height=0.35)
    elif name == 'b2':
        cfg = RobotConfig(mjcf_filename='b2/b2.xml', hip_height=0.485)
    elif 'hyqreal' in name:
        cfg = RobotConfig(mjcf_filename='hyqreal/hyqreal.xml', hip_height=0.5)
    else:
        raise ValueError(f'Unknown robot name: {robot_name}')

    return cfg

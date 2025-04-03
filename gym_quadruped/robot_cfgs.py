from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from omegaconf import OmegaConf


@dataclass
class RobotConfig:
    """Base class of a quadruped robot configuration."""

    urdf_filename: str
    hip_height: float  # height of the hip joint in normal stand pose
    qpos0_js: Optional[np.ndarray] = None  # Custom zero position of the joint space, if None defaults to the URDF.
    feet_geom_names: dict[str, str] = field(default_factory=lambda: {'FL': 'FL', 'FR': 'FR', 'RL': 'RL', 'RR': 'RR'})
    leg_joints: dict[str, list[str]] = field(
        default_factory=lambda: {
            'FL': ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint'],
            'FR': ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint'],
            'RL': ['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'],
            'RR': ['RR_hip_joint', 'RR_calf_joint', 'RR_thigh_joint'],
        }
    )
    # IMU related parameters _______________________________________________________
    accel_name: Optional[str] = None  # accelerometer name in the xml file
    gyro_name: Optional[str] = None  # gyroscope name in the xml file
    imu_site_name: Optional[str] = None  # imu site name in the xml file


@dataclass
class Go1Config(RobotConfig):  # noqa D101
    urdf_filename: str = 'go1.urdf'
    hip_height: float = 0.3


@dataclass
class Go2Config(RobotConfig):  # noqa D101
    urdf_filename: str = 'go2.urdf'
    hip_height: float = 0.28


@dataclass
class AliengoConfig(RobotConfig):  # noqa D101
    urdf_filename: str = 'aliengo.urdf'
    hip_height: float = 0.35


@dataclass
class B2Config(RobotConfig):  # noqa D101
    urdf_filename: str = 'b2.urdf'
    hip_height: float = 0.485


@dataclass
class HyqrealConfig(RobotConfig):  # noqa D101
    urdf_filename: str = 'hyqreal.urdf'
    hip_height: float = 0.5


@dataclass
class MiniCheetahConfig(RobotConfig):  # noqa D101
    urdf_filename: str = 'mini_cheetah.urdf'
    hip_height: float = 0.225
    qpos0_js: List[float] = np.concatenate((np.array([0, -np.pi / 2, 0] * 2), np.array([0, np.pi / 2, 0] * 2)))


def get_robot_config(robot_name: str) -> RobotConfig:
    """Get the robot configuration based on the robot name."""
    robot_configs = {
        'go1': Go1Config(),
        'go2': Go2Config(),
        'aliengo': AliengoConfig(),
        'b2': B2Config(),
        'hyqreal': HyqrealConfig(),
        'mini_cheetah': MiniCheetahConfig(),
    }

    if robot_name.lower() not in robot_configs:
        raise ValueError(f'Unknown robot name: {robot_name}')

    return robot_configs[robot_name.lower()]

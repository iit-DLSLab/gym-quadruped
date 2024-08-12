from __future__ import annotations

import xml.etree.ElementTree as xml_et

import numpy as np
import cv2
import noise
from pathlib import Path
from typing import List

from scipy.spatial.transform import Rotation


def list_to_str(vec: List[float]) -> str:
    """
    Converts a list of numbers to a space-separated string for URDF/XML formatting.

    Args:
        vec (List[float]): List of numbers.

    Returns:
        str: Space-separated string.
    """
    return " ".join(str(s) for s in vec)

def add_perlin_heightfield(
        asset: xml_et.Element,
        worldbody: xml_et.Element,
        position: List[float] = (0.0, 0.0, 0.0),  # position
        euler_xyz: List[float] = (0.0, 0.0, 0.0),  # attitude
        size: List[float] = (1.0, 1.0),  # width and length of the generated field in meters
        max_height: float = 0.4,    # Maximum height value of the heightfield in meters
        min_height: float = 0.001,  # Minimum height value of the heightfield in meters
        image_width: int = 128,  # height field image size
        img_height: int = 128,
        smooth: float = 100.0,  # smooth scale
        perlin_octaves: int = 6,  # perlin noise parameter
        perlin_persistence: float = 0.5,
        perlin_lacunarity: float = 2.0,
        output_hfield_image: str = "height_field.png") -> None:
    """
    Adds a Perlin noise-based heightfield to the scene.

    Args:
        asset (xml_et.Element): The asset element of the XML.
        worldbody (xml_et.Element): The worldbody element of the XML.
        position (List[float]): Position of the heightfield [x, y, z].
        euler_xyz (List[float]): Euler angles for the heightfield orientation [roll, pitch, yaw].
        size (List[float]): Size of the heightfield [width, length].
        max_height (float): Maximum height of the heightfield in meters.
        min_height (float): Height in the negative direction of the z axis in meters.
        image_width (int): Width of the heightfield image.
        img_height (int): Height of the heightfield image.
        smooth (float): Smoothing scale for Perlin noise.
        perlin_octaves (int): Number of octaves for Perlin noise. Higher values add more detail.
        perlin_persistence (float): Persistence value for Perlin noise. Controls amplitude of octaves.
        perlin_lacunarity (float): Lacunarity value for Perlin noise. Controls frequency of octaves.
        output_hfield_image (str): Filename for the output heightfield image.

    Octaves, Persistence, and Lacunarity:
        - Octaves: Successive layers of Perlin noise added together for complexity and detail.
        - Persistence: Controls the amplitude decrease of higher octaves (e.g., 0.5 reduces amplitude by half).
        - Lacunarity: Controls the frequency increase of higher octaves (e.g., 2.0 doubles the frequency).

    """
    # Ensure output directory exists
    output_path = Path(__file__).resolve().parent / f"assets"
    assert output_path.exists(), f"Output path {output_path.absolute().resolve()} does not exist."
    file_path = (output_path / output_hfield_image).with_suffix(".png")

    # Generating height field based on Perlin noise
    terrain_image = np.zeros((img_height, image_width), dtype=np.uint8)
    for y in range(image_width):
        for x in range(image_width):
            noise_value = noise.pnoise2(x / smooth, y / smooth,
                                        octaves=perlin_octaves,
                                        persistence=perlin_persistence,
                                        lacunarity=perlin_lacunarity)
            terrain_image[y, x] = int((noise_value + 1) / 2 * 255)

    cv2.imwrite(str(file_path.resolve()), terrain_image)

    hfield = xml_et.SubElement(asset, "hfield")
    hfield.attrib["name"] = "perlin_hfield"
    hfield.attrib["size"] = list_to_str([size[0] / 2.0, size[1] / 2.0, max_height, min_height])
    hfield.attrib["file"] = str(output_path.resolve())

    geo = xml_et.SubElement(worldbody, "geom")
    geo.attrib["type"] = "hfield"
    geo.attrib["hfield"] = "perlin_hfield"
    geo.attrib["pos"] = list_to_str(position)
    quat_xyzw = Rotation.from_euler("xyz", euler_xyz).as_quat(canonical=True)
    quat_wxyz = np.roll(quat_xyzw, 1)
    geo.attrib["quat"] = list_to_str(quat_wxyz)


# zyx euler angle to quaternion
def euler_to_quat(roll, pitch, yaw):
    cx = np.cos(roll / 2)
    sx = np.sin(roll / 2)
    cy = np.cos(pitch / 2)
    sy = np.sin(pitch / 2)
    cz = np.cos(yaw / 2)
    sz = np.sin(yaw / 2)

    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        dtype=np.float64,
    )


# zyx euler angle to rotation matrix
def euler_to_rot(roll, pitch, yaw):
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ],
        dtype=np.float64,
    )

    rot_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rot_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x


# 2d rotate
def rot2d(x, y, yaw):
    nx = x * np.cos(yaw) - y * np.sin(yaw)
    ny = x * np.sin(yaw) + y * np.cos(yaw)
    return nx, ny

# 3d rotate
def rot3d(pos, euler):
    R = euler_to_rot(euler[0], euler[1], euler[2])
    return R @ pos


# Add Box to scene
def add_box(asset: xml_et.Element,
            worldbody: xml_et.Element,
            position=[1.0, 0.0, 0.0],
            euler=[0.0, 0.0, 0.0], 
            size=[0.1, 0.1, 0.1]):
    geo = xml_et.SubElement(worldbody, "geom")
    geo.attrib["pos"] = list_to_str(position)
    geo.attrib["type"] = "box"
    geo.attrib["size"] = list_to_str(
        0.5 * np.array(size))  # half size of box for mujoco
    quat = euler_to_quat(euler[0], euler[1], euler[2])
    geo.attrib["quat"] = list_to_str(quat)


def add_world_of_boxes(model_file_path,
                       init_pos=[1.0, 0.0, 0.0],
                       euler=[0.0, -0.0, 0.0],
                       nums=[10, 10],
                       box_size=[0.5, 0.5, 0.1],
                       box_euler=[0.0, 0.0, 0.0],
                       separation=[0.2, 0.2],
                       box_size_rand=[1, 1, 1],
                       box_euler_rand=[0.2, 0.2, 0.2],
                       separation_rand=[0, 1],
                       random_roll_pitch=False):

    scene = xml_et.parse(model_file_path)
    root = scene.getroot()
    worldbody = root.find("worldbody")
    asset = root.find("asset")

    #local_pos = [0.0, 0.0, -0.5 * box_size[2]]
    local_pos = [0.0, 0.0, 0.0]
    new_separation = np.array(separation) + np.array(
        separation_rand) * np.random.uniform(-1.0, 1.0, 2)
    for i in range(nums[0]):
        local_pos[0] += new_separation[0]
        local_pos[1] = 0.0
        for j in range(nums[1]):
            new_box_size_xy = np.array(box_size)[0:2] + np.array(
                box_size_rand)[0:2] * np.random.uniform(-0.2, 0.2, 2)
            new_box_size_z = np.array(box_size)[2] + np.array(
                box_size_rand)[2] * np.random.uniform(-0.1, 0.15, 1)
            new_box_size = np.array([new_box_size_xy[0], new_box_size_xy[1], new_box_size_z[0]])
            
            if random_roll_pitch:
                new_box_euler = np.array(box_euler) + np.array(
                    box_euler_rand) * np.random.uniform(-1.0, 1.0, 3)
            else:
                new_box_euler = np.array(box_euler) 
                new_box_euler[2] = new_box_euler[2] + np.array(
                    box_euler_rand)[2] * np.random.uniform(-1, 1, 1)                
            
            new_separation_x = np.array(separation)[0] + np.array(
                separation_rand)[0] * np.random.uniform(0, 0.5, 1)
            new_separation_y = np.array(separation)[1] + np.array(
                separation_rand)[1] * np.random.uniform(-0.5, 0.5, 1)
            new_separation = np.array([new_separation_x[0], new_separation_y[0]])

            local_pos[1] += new_separation[1]
            pos = rot3d(local_pos, euler) + np.array(init_pos)
            add_box(asset, worldbody, pos, new_box_euler, new_box_size)
    return scene


def add_world_of_pyramid(model_file_path,
                    init_pos=[1.0, 0.0, 0.0],
                    yaw=0.0,
                    width=5,
                    max_height=0.15,
                    length=5,
                    stair_nums=5):
    scene = xml_et.parse(model_file_path)
    root = scene.getroot()
    worldbody = root.find("worldbody")
    asset = root.find("asset")

    local_pos = [0.0, 0.0, -0.05]
    height_rand = np.random.uniform(0.08, max_height, 1)
    stride_rand = np.random.uniform(0.5, 1.0, 1)
    for i in range(stair_nums):
        local_pos[2] += height_rand[0] 
        x, y = rot2d(local_pos[0], local_pos[1], yaw)
        new_width = width - stride_rand[0] * i
        new_length = length - stride_rand[0] * i
        add_box(asset, worldbody, [x + init_pos[0], y + init_pos[1], local_pos[2]],
                    [0.0, 0.0, yaw], [new_width, new_length, height_rand[0]])
        
    return scene
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
    quat = Rotation.from_euler("xyz", euler).as_quat(canonical=True, scalar_first=True)
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

    # variables to calculate the area of boxes
    max_abs_x = 0
    sign_x = 0  
    max_abs_y = 0
    sign_y = 0

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
            
            pos = Rotation.from_euler("xyz", euler).as_matrix() @ local_pos + np.array(init_pos)
            add_box(asset, worldbody, pos, new_box_euler, new_box_size)

            # calculate the area of boxes to take the maximum values and the center
            tmp_X = abs(local_pos[0] + init_pos[0])
            tmp_Y = abs(local_pos[1] + init_pos[1])
            if(tmp_X>=max_abs_x): 
                max_abs_x = tmp_X
                sign_x = 1 if tmp_X > 0 else -1
            if(tmp_Y>=max_abs_y): 
                max_abs_y = tmp_Y
                sign_y = 1 if tmp_Y > 0 else -1
            
    
    #apply sign to the absoulte max values
    max_x = max_abs_x * sign_x
    max_y = max_abs_y * sign_y

    # center of the area
    center = ((max_x + init_pos[0])/2, (max_y + init_pos[1])/2)

    # create a radius to spawn the robot at a safe distance
    if(max_abs_x >=max_abs_y): 
        radius = 1.2*np.sqrt(2*(max_x - center[0])*(max_x - center[0]))
    else:                      
        radius = 1.2*np.sqrt(2*(max_y - center[1])*(max_y - center[1]))
    
    return scene, radius, center


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

    # variables to calculate the area of boxes
    max_abs_x = 0
    sign_x = 0  
    max_abs_y = 0
    sign_y = 0

    local_pos = [0.0, 0.0, -0.05]
    height_rand = np.random.uniform(0.08, max_height, 1)
    stride_rand = np.random.uniform(0.5, 1.0, 1)
    for i in range(stair_nums):
        local_pos[2] += height_rand[0] 
        x, y, _ = Rotation.from_euler('xyz', [0, 0, yaw]).as_matrix() @ local_pos
        new_width = width - stride_rand[0] * i
        new_length = length - stride_rand[0] * i
        

        
        add_box(asset, worldbody, [x + init_pos[0], y + init_pos[1], local_pos[2]],
                    [0.0, 0.0, yaw], [new_width, new_length, height_rand[0]])
        
        # The first box - the one at the bottom - is the largest one
        if(i == 0):
            max_abs_x = abs(x + init_pos[0] + new_width/2.)
            max_abs_y = abs(y + init_pos[1] + new_length/2.)
            center = (x + init_pos[0], y + init_pos[1])
        

    # create a radius to spawn the robot at a safe distance
    if(max_abs_x >=max_abs_y): 
        radius = 1.2*np.sqrt(2*(max_abs_x - center[0])*(max_abs_x - center[0]))
    else:                      
        radius = 1.2*np.sqrt(2*(max_abs_y - center[1])*(max_abs_y - center[1]))
    
    
    return scene, radius, center
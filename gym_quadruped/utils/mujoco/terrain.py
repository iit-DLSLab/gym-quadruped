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



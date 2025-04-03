from __future__ import annotations

import os
from datetime import datetime

import cv2
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation


class Camera:
    """Camera class for capturing RGB images, depth images, and point clouds from a Mujoco simulation."""

    def __init__(self, width, height, fps, mj_model, mj_data, cam_name: str = '', save_dir='data/img/'):
        """Initialize the Camera object.

        Args:
            width (int): Width of the camera image.
            height (int): Height of the camera image.
            fps (float): Frames per second for the camera.
            mj_model (mj.MjModel): MuJoCo model object.
            mj_data (mj.MjData): MuJoCo data object.
            cam_name (str, optional): Name of the camera. Defaults to an empty string.
            save_dir (str, optional): Directory to save captured images. Defaults to 'data/img/'.
        """
        self._cam_name = cam_name
        self._mj_data = mj_model
        self._mj_model = mj_data
        self._save_dir = save_dir + self._cam_name + '/'
        self.save_counter = 0
        self.interval = float(1 / fps)
        self.last_time = float(mj_data.time)  # in seconds

        self._width = width
        self._height = height
        self._cam_id = self._mj_model.cam(self._cam_name).id

        self._renderer = mj.Renderer(self._mj_data, self._height, self._width)
        self._camera = mj.MjvCamera()
        self._scene = mj.MjvScene(self._mj_data, maxgeom=10_000)

        self._image = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        self._depth_plane = np.zeros((self._height, self._width, 1), dtype=np.float32)
        self._depth_image = np.zeros((self._height, self._width, 1), dtype=np.float32)
        self._seg_id_image = np.zeros((self._height, self._width, 3), dtype=np.float32)
        self._point_cloud = np.zeros((self._height, self._width, 1), dtype=np.float32)

        timestamp = str(datetime.now())
        timestamp = timestamp.replace(':', '_').replace(' ', '_')

        self.K = self.intrinsic_mat

        self._save_dir = os.path.join(self._save_dir, 'data_' + timestamp)

    @property
    def height(self) -> int:
        """Get the height of the camera.

        Returns:
                int: The height of the camera.
        """
        return self._height

    # ===================================================================
    @property
    def width(self) -> int:
        """Get the width of the camera.

        Returns:
                int: The width of the camera.
        """
        return self._width

    # ====================================================================
    @property
    def last_sim_time(self) -> float:
        """Get the last simulation time, in seconds, from camera function call.

        Returns:
                int: The last simulation time.
        """
        return self.last_time

    # ====================================================================
    @last_sim_time.setter
    def last_sim_time(self, time) -> None:
        """Set the last simulation time, in seconds, from camera function call.

        Returns:
                None
        """
        self.last_time = time

    # ====================================================================
    @property
    def save_dir(self) -> str:
        """Get the directory where images captured by the camera are saved.

        Returns:
                str: The directory where images captured by the camera are saved.
        """
        return self._save_dir

    # ====================================================================
    @property
    def name(self) -> str:
        """Get the name of the camera.

        Returns:
                str: The name of the camera.s
        """
        return self._cam_name

    # ====================================================================
    @property
    def intrinsic_mat(self) -> np.ndarray:
        """Compute the intrinsic camera matrix.

        Computes the camera matrix (K) based on the camera's field of view (fov), width (_width), and height
        (_height) parameters, following the pinhole camera model.

        Returns:
        np.ndarray: The intrinsic camera matrix (K), a 3x3 array representing the camera's intrinsic parameters.
        """
        # Convert the field of view from degrees to radians
        theta = np.deg2rad(self.fov)

        # Focal length calculation (f in terms of sensor width and height)
        f_x = (self._width / 2) / np.tan(theta / 2)
        f_y = (self._height / 2) / np.tan(theta / 2)

        # Pixel resolution (assumed to be focal length per pixel unit)
        alpha_u = f_x  # focal length in terms of pixel width
        alpha_v = f_y  # focal length in terms of pixel height

        # Optical center offsets (assuming they are at the center of the sensor)
        u_0 = (self._width - 1) / 2.0
        v_0 = (self._height - 1) / 2.0

        # Intrinsic camera matrix K
        K = np.array([[alpha_u, 0, u_0], [0, alpha_v, v_0], [0, 0, 1]])

        return K

    @property
    def frame_config(self) -> np.ndarray:
        """Compute the camera configuration (an homogeneous transformation matrix) in world coordinates.

        The transformation matrix is computed from the camera's position and orientation in world coordinates.
        The position and orientation are retrieved from the camera data.

        Returns:
        np.ndarray: The 4x4 homogeneous transformation matrix representing the camera's pose.
        """
        pos = self._mj_model.cam(self._cam_id).xpos
        rot = self._mj_model.cam(self._cam_id).xmat.reshape(3, 3).T
        T = np.eye(4)
        T[:3, :3] = Rotation.from_matrix(rot)
        T[:3, 3] = pos
        return T

    @property
    def projection_mat(self) -> np.ndarray:
        """Compute the projection matrix for the camera.

        The projection matrix is computed as the product of the camera's intrinsic matrix (K)
        and the homogeneous transformation matrix (T_world_cam).

        Returns:
        np.ndarray: The 3x4 projection matrix.
        """
        return self.intrinsic_mat @ self.frame_config

    @property
    def image(self) -> np.ndarray:
        """Return the captured RGB image."""
        self._renderer.update_scene(self._mj_model, camera=self.name)
        self._image = self._renderer.render()
        self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        return self._image

    @property
    def depth_image(self) -> np.ndarray:
        """Return the captured depth image."""
        self._renderer.update_scene(self._mj_model, camera=self.name)
        self._renderer.enable_depth_rendering()
        self._depth_plane = self._renderer.render()
        i_indices, j_indices = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing='ij')
        x_camera = (i_indices - self.K[0][2]) * self._depth_plane / self.K[0][0]
        y_camera = (j_indices - self.K[1][2]) * self._depth_plane / self.K[1][1]
        self._depth_image = np.sqrt(self._depth_plane**2 + x_camera**2 + y_camera**2)
        self._renderer.disable_depth_rendering()
        return self._depth_image

    @property
    def seg_image(self) -> np.ndarray:
        """Return the captured segmentation image based on object's id."""
        self._renderer.update_scene(self._mj_model, camera=self.name)
        self._renderer.enable_segmentation_rendering()

        self._seg_id_image = self._renderer.render()[:, :, 0].reshape((self.height, self.width))
        self._renderer.disable_segmentation_rendering()
        return self._seg_id_image

    @property
    def point_cloud(self) -> np.ndarray:
        """Return the captured point cloud."""
        self._point_cloud = self._depth_to_point_cloud(self.depth_image)
        return self._point_cloud

    @property
    def fov(self) -> float:
        """Get the field of view (FOV) of the camera.

        Returns:
        float: The field of view angle in degrees.
        """
        return self._mj_data.cam(self._cam_id).fovy[0]

    @property
    def id(self) -> int:
        """Get the identifier of the camera.

        Returns:
        int: The identifier of the camera.
        """
        return self._cam_id

    def _depth_to_point_cloud(self, depth_image: np.ndarray) -> np.ndarray:
        """Method to convert depth image to a point cloud in camera coordinates.

        Args:
        depth_image: The depth image we want to convert to a point cloud.

        Returns:
        np.ndarray: 3D points in camera coordinates.
        """
        # Get image dimensions
        dimg_shape = depth_image.shape
        height = dimg_shape[0]
        width = dimg_shape[1]

        # Create pixel grid
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

        # Flatten arrays for vectorized computation
        x_flat = x.flatten()
        y_flat = y.flatten()
        depth_flat = depth_image.flatten()

        # Negate depth values because z-axis goes into the camera
        depth_flat = -depth_flat

        # Stack flattened arrays to form homogeneous coordinates
        homogeneous_coords = np.vstack((x_flat, y_flat, np.ones_like(x_flat)))

        # Compute inverse of the intrinsic matrix K
        K_inv = np.linalg.inv(self.intrinsic_mat)

        # Calculate 3D points in camera coordinates
        points_camera = np.dot(K_inv, homogeneous_coords) * depth_flat

        # Homogeneous coordinates to 3D points
        points_camera = np.vstack((points_camera, np.ones_like(x_flat)))

        points_camera = points_camera.T

        # dehomogenize
        points_camera = points_camera[:, :3] / points_camera[:, 3][:, np.newaxis]

        return points_camera

    def shoot(self, autosave: bool = True, img=True, depth=True, seg=True) -> None:
        """Captures a new rgb image, depth image and point cloud from the camera.

        Args:
        autosave: If the camera rgb image, depth image and point cloud should be saved.
        img: If the rgb image should be saved.
        depth: If the depth image should be saved.
        seg: If the segmentation image should be saved.
        """
        self._image = self.image
        self._depth_image = self.depth_image
        self._point_cloud = self.point_cloud
        self._seg_image = self.seg_image
        if autosave:
            self.save(img=img, depth=depth, seg=seg)

    def save(self, img_name: str = '', img: bool = False, depth: bool = False, seg: bool = False) -> None:
        """Saves the captured image and depth information.

        Args:
        img_name: Name for the saved image file.
        img: If the rgb image should be saved.
        depth: If the depth image should be saved.
        seg: If the segmentation image should be saved.
        """
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)
            os.makedirs(os.path.join(self._save_dir, 'images'))

        ptr_string = 'saving '
        if img:
            ptr_string += 'rgb image '
        if depth:
            ptr_string += 'depth image '
        if seg:
            ptr_string += 'segmentation image '
        ptr_string += f'to {self.save_dir}'

        print(ptr_string)

        if img_name == '':
            if img:
                cv2.imwrite(
                    self._save_dir + f'img_{self.save_counter}.png',
                    cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR),
                )
            if seg:
                cv2.imwrite(self._save_dir + f'seg_{self.save_counter}.png', self.seg_image)
            if depth:
                np.save(self._save_dir + '/images/' + f'depth_{self.save_counter}.npy', self.depth_image)
            self.save_counter += 1
        else:
            if img:
                cv2.imwrite(
                    self._save_dir + f'{img_name}_rgb.png',
                    cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR),
                )
            if seg:
                cv2.imwrite(self._save_dir + f'{img_name}_seg.png', self.seg_image)
            if depth:
                np.save(self._save_dir + '/images/' + f'{img_name}_depth.npy', self.depth_image)

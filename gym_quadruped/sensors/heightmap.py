# Description: This module class is used to simulate an heightmap based on elevation data for mujoco simulation to be
# used inside the quadruped_pympc enviroment

# Authors:
# - Dessy Giovanni.

from __future__ import annotations

import copy

import mujoco
import mujoco.viewer
import numpy as np

np.set_printoptions(precision=3, suppress=True)


class HeightMap:
    """This class is used to create a heightmap using raycast sensors in mujoco."""

    def __init__(self, n, dist_x, dist_y, mj_model, mj_data):
        """Init function of the grid_map class.

        Args:
            n: is the size of the heightmap in particular the number of lines we want to have in our heightmap it must
            be an odd number but is handled by the constructor
            dist_x : distance between two consecutive points in the x direction
            dist_y : distance between two consecutive points in the y direction
            mj_model: mujoco model
            mj_data: mujoco data
        """
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.use_map_initialization = False
        self.last_time = float(mj_data.time)
        self.data = None

        self.geom_ids = -np.ones((n, n), dtype=np.int32)

        if n % 2 == 0:
            self.n = n + 1
        else:
            self.n = n
        self.dist_x = dist_x
        self.dist_y = dist_y
        self.sensor_matrix = np.empty((self.n, self.n, 1, 3))
        self.sensor_data_matrix = np.empty((self.n, self.n, 1, 3))
        self.old_sensor_data_matrix = np.empty((self.n, self.n, 1, 3))

    @property
    def last_sim_time(self) -> float:
        """Get the last simulation time, in seconds, from camera function call.

        Returns:
                int: The last simulation time.
        """
        return self.last_time

    @last_sim_time.setter
    def last_sim_time(self, time) -> None:
        """Set the last simulation time, in seconds, from camera function call.

        Returns:
                None
        """
        self.last_time = time

    def raycast_sensor(self, pos, dist):
        """This function is used to simulate a raycast sensor in mujoco.

        Args:
            pos: is the position of the sensor from where the ray is casted
            dist: is the distance between the sensors
            Requires a small hack of making the alpha of the feet 0.0 in the xml file

        Example:  RR_Calf : <geom size="0.0265" pos="0 0 -0.25" rgba="0 0 0 0" />
        """
        # In the XML, we usually set 1-2-3 for body related geom that we don't want to intersect
        self.geomgroup_test = np.array([1, 0, 0, 0, 1, 1], dtype=np.uint8)

        ray_sensor_site = np.array([pos[0], pos[1], pos[2] - dist], dtype=np.float64)  # Starting point of the ray
        direction_vector = np.array([0, 0, -1], dtype=np.float64)

        geomgroup = self.geomgroup_test
        # Flag indicating whether the ray should only intersect with static objects (1) or also with dynamic object (0)
        flg_static = 1
        bodyexclude = (
            -1
        )  # An optional parameter specifying a body ID to exclude from intersections. Use -1 to not exclude any body.
        geomid = np.zeros(1, dtype=np.int32)

        self.z = mujoco.mj_ray(
            m=self.mj_model,
            d=self.mj_data,
            pnt=ray_sensor_site,
            vec=direction_vector,
            geomgroup=geomgroup,
            flg_static=flg_static,
            bodyexclude=bodyexclude,
            geomid=geomid,
        )
        if self.use_map_initialization is True:
            intersection_point = ray_sensor_site + direction_vector * self.z if geomid[0] > 0 else [-1, -1, -1]
        else:
            intersection_point = ray_sensor_site + direction_vector * self.z
        return intersection_point
        # this is used to create the sensor matrix used to store the sensor data

    def create_sensor_matrix(self, center, yaw=0.0):
        """This is the main function used to create the grid map using the ray sensor data."""
        R_W2H = np.array([np.cos(yaw), np.sin(yaw), -np.sin(yaw), np.cos(yaw)], dtype=np.float64)
        R_W2H = R_W2H.reshape((2, 2))

        c = int((self.n - 1) / 2)
        # print(c)
        self.ref_robot = np.array([center[0], center[1], center[2] + 0.6], dtype=np.float64)
        self.sensor_matrix[c][c] = self.ref_robot

        # fill the elements based on a and b
        # the first loop is for the rows andthe second loop is for the columns
        # the x and y are filled base on a and b for the sensor position z is the same fr all sensors
        for i in range(self.n):
            p = c - i
            for j in range(self.n):
                k = c - j
                if i == j == c:
                    continue
                else:
                    # Take heightmap in the horizontal frame
                    offset = np.array([self.dist_x * p, self.dist_y * k], dtype=np.float64)
                    offset = R_W2H.T @ offset
                    self.grid_element_pos = np.array(
                        [
                            self.ref_robot[0] + offset[0],
                            self.ref_robot[1] + offset[1],
                            self.ref_robot[2],
                        ]
                    )

                    self.sensor_matrix[i][j] = self.grid_element_pos

        ####################################################################################################################
        # now i use ray cast to fill the sensor_data_matrix used to store the sensor data
        for i in range(self.n):
            for j in range(self.n):
                pos = [
                    self.sensor_matrix[i][j][0][0],
                    self.sensor_matrix[i][j][0][1],
                    self.sensor_matrix[i][j][0][2],
                ]
                dist = 0.07
                pos_data = self.raycast_sensor(pos, dist)

                self.grid_element_pos = np.array([pos_data[0], pos_data[1], pos_data[2]])
                self.sensor_data_matrix[i][j] = self.grid_element_pos
                if pos_data[2] == -1:
                    self.sensor_data_matrix[i][j] = self.old_sensor_data_matrix[i][j]
        self.old_sensor_data_matrix = copy.deepcopy(self.sensor_data_matrix)
        ##########################################################################################

        return self.sensor_data_matrix

    # this creates circular patches
    def circlecheck(self, a, b, x, y, r):  # noqa : D102
        c = (x - a) ** 2 + (y - b) ** 2 <= r**2
        d = 1 if c else 0
        return d

    # create rettangular patches for the vfa
    def rectanglecheck(self, a, b, x, y, r):
        """Check if the point (a, b) is inside the rectangle defined by the center (x, y) and radius r.

        Parameters:
            a (float): x-coordinate of the point.
            b (float): y-coordinate of the point.
            x (float): x-coordinate of the center of the rectangle.
            y (float): y-coordinate of the center of the rectangle.
            r (float): Radius from the center to the edges of the rectangle.

        Returns:
            int: 1 if the point is inside the rectangle, 0 otherwise.
        """
        rect_x1 = x - r
        rect_x2 = x + r
        rect_y1 = y - r
        rect_y2 = y + r
        if rect_x1 <= a <= rect_x2 and rect_y1 <= b <= rect_y2:
            return 1
        else:
            return 0

    def update_height_map(self, center, yaw):  # noqa: D102
        self.data = self.create_sensor_matrix(center, yaw)
        return self.data

    def create_height_map(self):  # noqa: D102
        self.data = self.create_sensor_matrix(self.mj_data.xpos[1])

        return self.data

    def get_height(self, target):
        """This function is used to get the height of the terrain at the foot position."""
        if self.data is not None:
            nearest_datapoint = np.array([0, 0, 0])
            for i in range(self.n):
                for j in range(self.n):
                    if np.linalg.norm(self.data[i][j][0][0:2] - target[0:2]) < np.linalg.norm(
                        nearest_datapoint[0:2] - target[0:2]
                    ):
                        nearest_datapoint = self.data[i][j][0] + 0.02
            return nearest_datapoint[2]

        return None

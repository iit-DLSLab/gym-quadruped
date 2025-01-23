'''
Copyright (c) 2025 Hilton-Marques <https://my.github.com/Hilton-Marques>

Created Date: Saturday, January 18th 2025, 4:41:25 pm
Author: Hilton-Marques

Description: IMU class which is a wrapper for the accelerometer and
gyroscope defined in Mujoco
HISTORY:
Date      	By	Comments
----------	---	----------------------------------------------------------
'''

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

class IMU:
  def __init__(self,
               mj_model,
               mj_data,
               accel_name,
               gyro_name,
               imu_site_name,
               acc_noise: float = 0.01,
               gyro_noise: float = 0.01,
               acc_bias: float = 0.01,
               gyro_bias: float = 0.01):
      '''
      Imu constructor

      mj_model: mujoco model
      mj_data: mujoco data
      accel_name: accelerometer name in the xml file
      gyro_name: gyroscope name in the xml file
      imu_site_name: imu site name in the xml file
      acc_noise: float with accelerometer's std assuming the same for all axes (m/s²)
      gyro_noise: float with gyroscope's std assuming the same for all axes (rad/s)
      acc_bias: float with accelerometer's bias assuming the same for all axes (m/s³)
      gyro_bias: float with gyroscope's bias assuming the same for all axes (rad/s²)
      '''
      self._mj_model = mj_model
      self._mj_data = mj_data

      self._accel_noise = acc_noise
      self._gyro_noise = gyro_noise
      self._accel_bias_drift = acc_bias
      self._gyro_bias_drift = gyro_bias

      self._accel_name = accel_name
      self._gyro_name = gyro_name

      self._accel_id = self._GetSensorId(accel_name)
      self._gyro_id = self._GetSensorId(gyro_name)
      self._imu_site_id = self._GetSiteId(imu_site_name)
      
      # Store the bias of the IMU
      self._imu_gyro_bias = np.zeros(3)
      self._imu_accel_bias = np.zeros(3)

      # Build IMU frame w.r.t to the base frame
      # this is a geometric transformation from frame b 
      # to frame i
      self.i_X_b = self._BuildIMUFrame()

      # Turn off to inspect data
      self._show = False

#=============================================================================== 
  def GetAccel(self, dt = 1.0) -> np.array:
    '''
    Get current linear acceleration measured by the IMU
    '''
    base_lin_acc_noise = np.random.normal(0, self._accel_noise, 3) 

    # Assuming dt*drift = bias (TODO: pass dt)
    self._imu_accel_bias += np.random.normal(0, self._accel_bias_drift, 3)

    accel_id = self._accel_id
    accel = self._mj_data.sensordata[accel_id:accel_id + 3] 
    # another option is with 
    #accel = self._mj_data.sensor(self._accel_name).data
    # but I believe it is faster to use the sensordata

    # add noise and biases to the real value
    accel +=  dt * self._imu_accel_bias + base_lin_acc_noise

    return accel, base_lin_acc_noise, self._imu_accel_bias

#=============================================================================== 
  def GetGyro(self, dt = 1.0) -> np.array:
    '''
    Get current angular velocity measured by the IMU
    '''
    base_ang_vel_noise = np.random.normal(0, self._gyro_noise, 3) 
    self._imu_gyro_bias  = self._imu_gyro_bias + np.random.normal(0, self._gyro_bias_drift, 3)

    gyro_id = self._gyro_id
    gyro = self._mj_data.sensordata[gyro_id:gyro_id + 3]
    # add noise and biases to the real value
    gyro +=  dt * self._imu_accel_bias + base_ang_vel_noise

    return gyro, base_ang_vel_noise, self._imu_gyro_bias
  
#=============================================================================== 
  def _GetSensorId(self, sensor_name) -> int:
    return self._mj_model.sensor(name=sensor_name).id

#=============================================================================== 
  def _BuildIMUFrame(self) -> np.array:
    '''
    Build IMU frame w.r.t to the base frame. This should be called only once
    '''
    # Build Inverse of IMU frame
    imu_pos = self._mj_data.site(self._imu_site_id).xpos
    imu_frame = self._mj_data.site(self._imu_site_id).xmat.reshape(3, 3)
    imu_frame = Rotation.from_matrix(imu_frame)

    # Create inverse of the IMU frame
    X_imu = np.eye(4)
    X_imu[0:3, 0:3] = imu_frame.as_matrix()
    X_imu[0:3, 3] = imu_pos

    # Build Body frame
    com_pos = self._mj_data.qpos[0:3]  # world frame
    quat_wxyz = self._mj_data.qpos[3:7]  # world frame (wxyz) mujoco convention
    quat_xyzw = np.roll(quat_wxyz, -1)  # SciPy convention (xyzw)
    X_B_inv = np.eye(4)
    X_B_inv[0:3, 0:3] = Rotation.from_quat(quat_xyzw).as_matrix().T
    X_B_inv[0:3, 3] = -np.dot(X_B_inv[0:3, 0:3], com_pos)

    X = np.dot(X_B_inv, X_imu)

    return X

#=============================================================================== 
  def GetIMUFrame(self) -> np.array:
    '''
    Get IMU frame w.r.t to the base frame
    '''
    return self.i_X_b

#=============================================================================== 
  def PrepareToShow(self):
    '''
    Create canvas and legends for the IMU data
    '''
    self.time = []
    self.accel_x = []
    self.accel_bias_x = []
    self.gyro_x = []
    self.gyro_bias_x = []
    plt.ion()  # Turn on interactive mode
    fig, axs = plt.subplots(2, 1)
    
    # Plot for white noise
    accel_line_noise, = axs[0].plot([], [], label="Accel noise x")
    gyro_line_noise, = axs[0].plot([], [], label="Gyro noise x")
    axs[0].set_xlabel("Time (steps)")
    axs[0].set_ylabel("white noise")
    axs[0].legend()

    # Plot for bias noise
    accel_line_bias, = axs[1].plot([], [], label="Accel bias x")
    gyro_line_bias, = axs[1].plot([], [], label="Gyro bias x")
    axs[1].set_xlabel("Time (steps)")
    axs[1].set_ylabel("brownian noise")
    axs[1].legend()

    # Store relevant variables
    self.ax_white_noise = axs[0]
    self.ax_bias_noise = axs[1]
    self._show = True
    self.accel_line_noise_x = accel_line_noise
    self.gyro_line_noise_x = gyro_line_noise
    self.accel_line_bias_x = accel_line_bias
    self.gyro_line_bias_x = gyro_line_bias

#=============================================================================== 
  def Show(self, time, accel_noise, gyro_noise, accel_bias, gyro_bias):
    '''
    Show the IMU data in the canvas
    '''
    if (not self._show):
      return

    self.time.append(time)
    self.accel_x.append(accel_noise[0])
    self.gyro_x.append(gyro_noise[0])
    self.accel_bias_x.append(accel_bias[0])
    self.gyro_bias_x.append(gyro_bias[0])

    # Update white noise
    self.accel_line_noise_x.set_xdata(self.time)
    self.accel_line_noise_x.set_ydata(self.accel_x)
    self.gyro_line_noise_x.set_xdata(self.time)
    self.gyro_line_noise_x.set_ydata(self.gyro_x)
    self.ax_white_noise.relim()
    self.ax_white_noise.autoscale_view()
    plt.draw()
    plt.pause(0.001)

    # Update bias noise
    self.accel_line_bias_x.set_xdata(self.time)
    self.accel_line_bias_x.set_ydata(self.accel_bias_x)
    self.gyro_line_bias_x.set_xdata(self.time)
    self.gyro_line_bias_x.set_ydata(self.gyro_bias_x)
    self.ax_bias_noise.relim()
    self.ax_bias_noise.autoscale_view()
    plt.draw()
    plt.pause(0.001)

#=============================================================================== 
  def _GetSiteId(self, site_name) -> int:
    site_id = self._mj_model.site(name=site_name).id

    return site_id

"""Copyright (c) 2025 Hilton-Santana <https://my.github.com/Hilton-Santana>.

Created Date: Saturday, January 25th 2025, 6:24:11 pm
Author: Hilton-Santana hiltonmarquess@gmail.com

Description: Class for collecting proprioceptive and ground truth data
from the robot and saving it in a hdf5 file.
HISTORY:
Date      	By	Comments
----------	---	----------------------------------------------------------
"""

import json
from pathlib import Path

import h5py
import numpy as np

from gym_quadruped.quadruped_env import QuadrupedEnv


def save_dict_to_h5(h5group, data):
	"""
	Recursively save a nested dictionary to an HDF5 group.

	h5group: An open h5py.Group object.
	data: Dictionary to save.
	"""
	for key, value in data.items():
		if isinstance(value, dict):  # If value is a dict, create a subgroup and recurse
			subgroup = h5group.require_group(key)
			save_dict_to_h5(subgroup, value)
		elif isinstance(value, (list, tuple)):  # Convert lists/tuples to JSON strings
			h5group.attrs[key] = json.dumps(value)
		elif isinstance(value, (str, int, float, bool, np.ndarray)):  # Store primitive types
			h5group.attrs[key] = value
		elif value is None:  # Store None as a special case
			pass
		else:
			raise TypeError(f"Cannot save type {type(value)} for key '{key}'")


def load_dict_from_h5(h5group):
	"""
	Recursively load a nested dictionary from an HDF5 group.

	h5group: An open h5py.Group object.
	"""
	data = {}

	# Load attributes (key-value pairs)
	for key, value in h5group.attrs.items():
		try:
			data[key] = json.loads(value)  # Decode JSON strings back to lists/tuples
		except (json.JSONDecodeError, TypeError):
			data[key] = value  # Otherwise, store as-is

	# Load nested groups (dictionaries)
	for key, subgroup in h5group.items():
		if isinstance(subgroup, h5py.Group):
			data[key] = load_dict_from_h5(subgroup)

	return data


class H5Writer:
	def __init__(self, file_path, env: QuadrupedEnv):
		"""
		Initialize the H5Writer object.

		Parameters:
		- file_path: str, path to the HDF5 file.
		- observation_space: gym.spaces.Dict, observation space of the QuadrupedEnv, used to define which observations are stored.
		- env_hparams: arguments used in the QuadrupedEnv constructor, used for reproducing the env used in data collection.
		"""
		self.file_path = Path(file_path)
		self.file_path.parent.mkdir(parents=True, exist_ok=True)

		with h5py.File(self.file_path, 'w') as hf:  # Open file in write mode
			# Save environment hyperparameters for reproducibility
			save_dict_to_h5(hf.create_group('env_hparams'), env.get_hyperparameters())

			# Create a group for recordings
			recordings = hf.create_group('recordings')

			# Create dataset for each data type (proprioceptive)
			recordings.create_dataset('time', shape=(0, 0, 1), maxshape=(None, None, 1), dtype='float64')
			for key, space in env.observation_space.spaces.items():
				shape = (0, 0) + space.shape  # Trajectory/Episode id, time, obs_shape
				max_shape = (None, None) + space.shape  # Trajectory/Episode id, time, obs_shape
				recordings.create_dataset(key, shape=shape, maxshape=max_shape, dtype='float64')
			# Action space
			shape = (0, 0) + env.action_space.shape  # Trajectory/Episode id, time, action_shape
			max_shape = (None, None) + env.action_space.shape  # Trajectory/Episode id, time, action_shape
			recordings.create_dataset('action', shape=shape, maxshape=max_shape, dtype='float64')

	def append_trajectory(self, state_obs_traj: dict[str, np.ndarray], time: np.ndarray):
		"""
		Append a trajectory/episode to the dataset.

		Parameters:
		- state_obs_traj: dict[str, np.ndarray], where each key is an observation name
		                  and the value is a NumPy array of shape (T, *obs_shape).
		- time: np.ndarray of shape (T, 1) representing the time steps.
		"""
		num_steps = time.shape[0]

		# Ensure all observations have the same first dimension (T)
		for key, value in state_obs_traj.items():
			if value.shape[0] != num_steps:
				raise ValueError(
					f'Observation {key} has inconsistent time steps: {value.shape[0]} vs {num_steps} in time array.'
				)

		with h5py.File(self.file_path, 'a') as hf:  # Open file in append mode
			recordings = hf['recordings']

			# Find the next available trajectory index
			current_trajectories = recordings['time'].shape[0]  # Number of stored trajectories
			new_traj_idx = current_trajectories

			# Resize the time dataset to accommodate the new trajectory
			recordings['time'].resize((new_traj_idx + 1, num_steps, 1))
			recordings['time'][new_traj_idx, :, :] = time  # Store new trajectory's time data

			# Resize and store each observation dataset
			for key, value in state_obs_traj.items():
				obs_dataset = recordings[key]
				obs_shape = value.shape[1:]  # Extract shape after time dimension

				# Resize dataset to add the new trajectory
				obs_dataset.resize((new_traj_idx + 1, num_steps) + obs_shape)
				obs_dataset[new_traj_idx, :, ...] = value  # Store new observation data


class H5Reader:
	def __init__(self, file_path):
		self.file_path = file_path
		# Open file
		file_path = Path(file_path)
		assert file_path.exists(), f'File not found: {file_path.absolute()}'
		self.h5py_file = h5py.File(self.file_path, 'r')

		# Check if file exists
		if not self.h5py_file:
			raise Exception('File not found')

		# Get groups
		self.recordings = self.h5py_file['recordings']
		self.env_hparams = load_dict_from_h5(self.h5py_file['env_hparams'])

		# Get number of trajectories
		self.n_trajectories = self.recordings['time'].shape[0]

	def len(self):
		return self.n_trajectories

	def get_trajectory(self, traj_idx):
		# Get data
		time = self.get(self.recordings, 'time', traj_idx, 1)
		traj_data = {}
		for key in self.recordings:
			if key != 'time':
				traj_data[key] = self.get(self.recordings, key, traj_idx, self.recordings[key].shape[2:])

		return time, traj_data

	def get(self, group, key, traj_idx, obs_shape):
		return group[key][traj_idx, :, ...]

	def close(self):
		self.h5py_file.close()

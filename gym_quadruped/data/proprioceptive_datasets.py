# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 17/02/25
from os import PathLike
from pathlib import Path

import numpy as np
import torch
from gym_quadruped.utils.data.h5py import H5Reader
from torch.utils.data import Dataset


class ProprioceptiveDataset(Dataset):
    """Dataset for classification/regression tasks using proprioceptive data.

    Args:
        data_file: (Path) Path to the HDF5 file containing the data to be read by a gym_quadruped.utils.data.h5_dataset.H5Reader.
            Dataset is assumed to be composed of observations queried by name and of shape (n_time_frames, n_features).
        x_obs_names: (list[str]) List of the names of the observations to be used as input features.
        y_obs_names: (list[str]) List of the names of the observations to be used as output features.
        x_frames: (int) Number of time frames to be used as input features.
        y_frames: (int) Number of time frames to be used as output features.
        mode: (str) If "dynamic" x and y are of the form `x = [t, t-1, ..., t-x_frames]` and `y = [t+1, t+2, ..., t+y_frames]`.
            If "static" x and y are of the form `x = [t-x_frames, ...,t-1, t]` and `y = [t-y_frames, ...,t-1, t]`.
        load_to_memory: (bool) If True, the dataset is loaded to memory for faster access.
        dtype: (torch.dtype) Data type of the dataset.
        device: (torch.device) Device to load the dataset to.
    """

    def __init__(
        self,
        data_file: PathLike,
        x_obs_names,
        y_obs_names,
        x_frames: int = 1,
        y_frames: int = 1,
        mode='static',  # "static" | "dynamic"
        load_to_memory=False,
        dtype=torch.float32,
        device=None,
    ):
        assert x_frames > 0 and y_frames > 0, 'X and Y need to be composed of at least one frame.'
        self.x_frames, self.y_frames = x_frames, y_frames
        # Load the Gym Quadruped dataset.
        self.h5file = H5Reader(data_file)
        for obs_name in x_obs_names + y_obs_names:
            assert obs_name in self.h5file.recordings.keys(), (
                f'Observation {obs_name} not in {self.h5file.recordings.keys()}'
            )

        self.x_obs_names, self.y_obs_names = x_obs_names, y_obs_names
        self.device = device  # Device to load the dataset to
        self.dtype = dtype
        self.mean_vars = {}  # Mean and variance of each observation in the dataset

        self._mode = mode
        self._load_to_memory = load_to_memory  # Load dataset to RAM / Device
        self._n_samples = None
        self._traj_lengths = {}  # Dataset can be composed of trajectories/episodes of different lengths
        self._indices = []  # Indices of samples in the raw_data

        self.compute_sample_indices()
        self._memory_data = None
        if self._load_to_memory:
            self._load_dataset_to_memory()

    def compute_sample_indices(self):
        """Compute the indices of the samples in the dataset.

        Dataset is composed of trajectories of shape (n_time_frames, n_features).
        The indices are tuples (traj_id, slice_idx) where slice_idx is a slice object indicating the start and end of
        the sample indices in time for the trajectory with id traj_id.
        """
        tmp_obs_name = self.x_obs_names[0]
        if self._mode == 'static':
            context_length = max(self.x_frames, self.y_frames)  #
        elif self._mode == 'dynamic':
            context_length = self.x_frames + self.y_frames
        else:
            raise ValueError(f"Mode {self._mode} not supported. Choose 'static' or 'dynamic'.")

        for traj_id in range(self.h5file.n_trajectories):
            traj_len = self.h5file.recordings[tmp_obs_name][traj_id].shape[0]
            traj_slices = self._slices_from_traj_len(traj_len, context_length, time_lag=1)
            self._indices.extend([(traj_id, s) for s in traj_slices])
            self._traj_lengths[traj_id] = traj_len

            for obs_name in self.x_obs_names + self.y_obs_names:
                try:
                    assert self.h5file.recordings[obs_name][traj_id].shape[0] == traj_len, (
                        f'Obs {tmp_obs_name} and {obs_name} have different time dimensions for trajectory {traj_id}.'
                    )
                except Exception as e:
                    raise Exception(f'Unable to assert {obs_name} for traj_id={traj_id}') from e

    @property
    def n_trajectories(self):
        """Returns the number of trajectories in the dataset."""
        return len(self._traj_lengths)

    @property
    def raw_data(self):
        """Returns the raw data contained in the dataset."""
        if self._load_to_memory:
            return self._memory_data
        else:
            return self.h5file.recordings

    @property
    def numpy_arrays(self):
        """Returns the raw data contained in the dataset."""
        if self._load_to_memory:
            all_data = self._memory_data
            np_data = {}
            for obs_name, traj_list in all_data.items():
                np_data[obs_name] = [traj for id, traj in enumerate(traj_list) if id in self._traj_lengths]
            return np_data
        else:
            raise ValueError('Raw data is not loaded to memory. Use `load_to_memory=True` to access the raw data.')

    def _load_dataset_to_memory(self):
        """Loads the dataset to memory for faster access."""
        self._memory_data = {}
        for obs_name in self.x_obs_names + self.y_obs_names:
            obs_data = []  # Trajectories might have different lengths
            for traj_id in range(self.h5file.n_trajectories):
                traj_data = self.h5file.recordings[obs_name][traj_id]
                obs_data.append(torch.tensor(traj_data).to(device=self.device, dtype=self.dtype))
            self._memory_data[obs_name] = obs_data

    def shuffle(self, seed=None):
        """Shuffles the dataset."""
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(self._indices)

    def __getitem__(self, idx):
        """Return x in the past and y in the future for the idx-th sample in the dataset.

        Args:
            idx: (int) Index of the sample in the dataset.

        Returns:
            x_obs: (dict[str, ArrayLike]) input observations with shape (x_frames, obs_dim) per observation name in `x_obs_names`.
            y_obs: (dict[str, ArrayLike]) output observations with shape (y_frames, obd_dim) per observation name in `y_obs_names`.
        """
        traj_idx, window_slice = self._indices[idx]
        if self._mode == 'static':
            x_slice = slice(-self.x_frames, None)
            y_slice = slice(-self.y_frames, None)
        elif self._mode == 'dynamic':
            x_slice = slice(0, self.x_frames)
            y_slice = slice(-self.y_frames, None)

        x_obs, y_obs = {}, {}

        try:
            for obs_name in self.x_obs_names:  # X is composed of the first x_frames observations
                x_obs[obs_name] = self.raw_data[obs_name][traj_idx][window_slice][x_slice]
            for obs_name in self.y_obs_names:  # Y is composed of the last y_frames observations
                y_obs[obs_name] = self.raw_data[obs_name][traj_idx][window_slice][y_slice]
        except Exception as e:
            raise Exception(
                f'Error fetching {obs_name} from traj={traj_idx} slice={window_slice}. '
                f'This traj has shape {self.raw_data[obs_name][traj_idx].shape}'
            ) from e
        return x_obs, y_obs

    def compute_obs_moments(self, obs_reps: dict = None):
        """Computes the mean and variance for each observation in x_obs_names and y_obs_names."""
        for obs_name in self.x_obs_names + self.y_obs_names:
            trajs = [self.h5file.recordings[obs_name][traj_id] for traj_id in self._traj_lengths.keys()]
            obs_data = np.concatenate(trajs, axis=0, dtype=np.float32)
            if obs_reps is not None:
                from symmetric_learning.nn.symmetric.stats import var_mean

                obs_var, obs_mean = var_mean(torch.tensor(obs_data), obs_reps[obs_name])
            else:
                obs_mean = np.mean(obs_data, axis=0)
                obs_var = np.var(obs_data, axis=0)
            self.mean_vars[obs_name] = (obs_mean, obs_var)

    def subset_dataset(self, trajectory_ids) -> 'ProprioceptiveDataset':
        """Creates a subset of the dataset containing only the specified trajectories."""
        assert len(trajectory_ids) > 0, 'Trajectory ids must be a non-empty list.'

        subset = ProprioceptiveDataset(
            self.h5file.file_path,
            self.x_obs_names,
            self.y_obs_names,
            self.x_frames,
            self.y_frames,
            mode=self._mode,
            load_to_memory=self._load_to_memory,
            dtype=self.dtype,
            device=self.device,
        )

        # Filter indices and trajectory lengths
        subset._indices = [idx for idx in self._indices if idx[0] in trajectory_ids]
        for i in range(self.h5file.n_trajectories):
            if i not in trajectory_ids:
                subset._traj_lengths.pop(i)

        return subset

    def __len__(self):
        return len(self._indices)

    @staticmethod
    def _slices_from_traj_len(time_horizon: int, context_length: int, time_lag: int) -> list[slice]:
        """Returns the list of slices (start_time_idx, end_time_idx) for each context window in the trajectory.

        Args:
            time_horizon: (int) Number time-frames of the trajectory.
            context_length: (int) Number of time-frames per context window
            time_lag: (int) Time lag between successive context windows.

        Returns:
            list[slice]: List of slices for each context window.

        Examples:
        --------
        >>> time_horizon, context_length, time_lag = 10, 4, 2
        >>> slices = TimeSeriesDataset._slices_from_traj_len(time_horizon, context_length, time_lag)
        >>> for s in slices:
        ...     print(f'start: {s.start}, end: {s.stop}')
        start: 0, end: 4
        start: 2, end: 6
        start: 4, end: 8
        start: 6, end: 10

        """
        slices = []
        for start in range(0, time_horizon - context_length + 1, time_lag):
            end = start + context_length
            slices.append(slice(start, end))

        return slices

    def __repr__(self):
        return f'{len(self._traj_lengths)} trajectories and {len(self)} total samples.'


if __name__ == '__main__':
    data_path = Path('aliengo/proprioceptive_data_ep=10_steps=1999.h5').absolute()

    dataset = ProprioceptiveDataset(
        data_path,
        x_obs_names=['qpos_js', 'qvel_js'],
        y_obs_names=['imu_acc', 'imu_gyro'],
        x_frames=10,
        y_frames=1,
        mode='static',
    )
    print(len(dataset))
    for i in range(10):
        x, y = dataset[i]
        for obs_name, obs_val in x.items():
            print(f'X: {obs_name}: {np.asarray(obs_val).shape}')
        for obs_name, obs_val in y.items():
            print(f'Y: {obs_name}: {np.asarray(obs_val).shape}')

    # _______________

    dataset = ProprioceptiveDataset(
        data_path,
        x_obs_names=['qpos_js', 'qvel_js'],
        y_obs_names=['imu_acc', 'imu_gyro'],
        x_frames=10,
        y_frames=5,
        mode='dynamic',
    )
    print(len(dataset))
    for i in range(10):
        x, y = dataset[i]
        for obs_name, obs_val in x.items():
            print(f'X: {obs_name}: {np.asarray(obs_val).shape}')
        for obs_name, obs_val in y.items():
            print(f'Y: {obs_name}: {np.asarray(obs_val).shape}')

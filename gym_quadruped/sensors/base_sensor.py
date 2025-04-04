import numpy as np


class Sensor:
    """Base class for all sensors in the environment."""

    def __init__(self, mj_model, mj_data, **kwargs):
        # Private variables as the Sensor user should avoid modifying them
        self._mj_model = mj_model
        self._mj_data = mj_data

    def step(self, **kwargs) -> None:
        """Simulate the sensor measurement process.

        This function will be called at every simulation step by the
        environment. The user should not call this function.

        Args:
            **kwargs: Additional arguments to pass to the sensor step function.
        """
        raise NotImplementedError

    def get_observation(self, obs_name: str) -> np.ndarray:
        """Get observation from the sensor.

        Args:
            obs_name: Name of the observation to get (e.g., force, force noise)

        Returns:
            np.ndarray: Observation value
        """
        raise NotImplementedError

    @staticmethod
    def available_observations() -> list[str]:
        """Get all available observations from the sensor.

        Returns:
            List[str]: List of available observation names
        """
        raise NotImplementedError

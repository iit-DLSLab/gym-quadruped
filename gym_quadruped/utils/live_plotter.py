import matplotlib as mpl

mpl.use('TkAgg')
import contextlib
import multiprocessing as mp
import os
import random
import signal
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class MujocoPlotter:
    """TODO: DOCSTRING."""

    def __init__(self, enable=True):
        self.plots = {}

        self.legs = ['FL', 'FR', 'RL', 'RR']
        self.joint_names = ['HAA', 'HFE', 'KFE']
        self.predefined_plots = ['Torque', 'JointPos', 'JointVel', 'FootContacts', 'LinAcc', 'AngVel']
        self.axis = ['X', 'Y', 'Z']

        self.all_plot_enable = enable

    def create(
        self,
        figure_name: str,
        subplot_titles: list,
        y_limits: list = None,
        rows: int = 1,
        cols: int = 1,
        window_size: int = 50,
        plots_per_ax: int = 1,
    ):
        """Create new Plot figure."""
        if y_limits is None:
            y_limits = [-1, 1]
        plotter = MultiLivePlotter(
            figure_name=figure_name,
            num_subplots=rows * cols,
            subplot_titles=subplot_titles,
            nrows=rows,
            ncols=cols,
            window_size=window_size,
            x_limits=[(0, window_size)] * (rows * cols),
            y_limits=y_limits * (rows * cols),
            plot_per_ax=plots_per_ax,
        )
        self.plots[figure_name] = plotter

    # ---------------------------------------------------
    def predefined_plot(  # noqa: D102
        self, name: str, y_limit: list, legs: list = None, joint_names: list = None, window_size: int = 50
    ):
        if name not in self.predefined_plots:
            print(f'Error: predefined plot {name} does not exist between: {self.predefined_plots}')
            return

        if legs is None:
            legs = self.legs
        if joint_names is None:
            joint_names = self.joint_names

        titles = []
        rows = 0
        cols = 0
        for leg in legs:
            rows += 1
            cols = 0
            for joint_name in joint_names:
                cols += 1
                titles += [f'{name} {leg}_{joint_name}']
        self.create(
            figure_name=name,
            rows=rows,
            cols=cols,
            window_size=window_size,
            subplot_titles=titles,
            y_limits=y_limit,
        )
        return legs, joint_names

    def torque_plot(self, legs: list = None, joint_names: list = None, window_size: int = 50, enable: bool = True):  # noqa: D102
        if enable is False or self.all_plot_enable is False:
            return
        self.torque_legs, self.torque_joint_names = self.predefined_plot(
            name='Torque',
            y_limit=[(-120, 120)],
            legs=legs,
            joint_names=joint_names,
            window_size=window_size,
        )

    def jointpos_plot(self, legs: list = None, joint_names: list = None, window_size: int = 50, enable: bool = True):  # noqa: D102
        if enable is False or self.all_plot_enable is False:
            return
        self.jp_legs, self.jp_joint_names = self.predefined_plot(
            name='JointPos',
            y_limit=[(-3.5, 3.5)],
            legs=legs,
            joint_names=joint_names,
            window_size=window_size,
        )

    def jointvel_plot(self, legs: list = None, joint_names: list = None, window_size: int = 50, enable: bool = True):  # noqa: D102
        if enable is False or self.all_plot_enable is False:
            return
        self.jv_legs, self.jv_joint_names = self.predefined_plot(
            name='JointVel',
            y_limit=[(-15, 15)],
            legs=legs,
            joint_names=joint_names,
            window_size=window_size,
        )

    def footContact_plot(self, legs: list = None, joint_names: list = None, window_size: int = 50, enable: bool = True):  # noqa: D102
        if enable is False or self.all_plot_enable is False:
            return
        self.contact_legs, _ = self.predefined_plot(
            name='FootContacts',
            y_limit=[(-0.8, 1.2)],
            legs=legs,
            joint_names=[' '],
            window_size=window_size,
        )

    def lin_acc_plot(self, axis: list = None, window_size: int = 50, enable: bool = True):  # noqa: D102
        if enable is False or self.all_plot_enable is False:
            return

        if axis is None:
            axis = self.axis

        _, self.lin_acc = self.predefined_plot(
            name='LinAcc',
            y_limit=[(-5, 13)],
            legs=['trunk'],
            joint_names=axis,
            window_size=window_size,
        )

    def ang_vel_plot(self, axis: list = None, window_size: int = 50, enable: bool = True):  # noqa: D102
        if enable is False or self.all_plot_enable is False:
            return

        if axis is None:
            axis = self.axis

        _, self.ang_vel = self.predefined_plot(
            name='AngVel',
            y_limit=[(-5, 5)],
            legs=['Trunk'],
            joint_names=axis,
            window_size=window_size,
        )

    def predefine_update(self, name, data, selected_legs, selected_joints, legs_attr=False):
        """TODO: Docstring required."""
        if legs_attr:
            data = [value for arr in data.to_list() for value in arr]  # convert LegArray to list

        if name not in self.predefined_plots:
            print(f'Error: predefined plot {name} does not exist between: {self.predefined_plots}')
            return

        if len(data) == 12:
            # Data corresponds to [FL,FR,RL,RR] x [HAA,HFE,KFE]
            filtered_values = [
                data[i * 3 + j]
                for i, leg in enumerate(self.legs)
                for j, joint in enumerate(self.joint_names)
                if leg in selected_legs and joint in selected_joints
            ]
        elif len(data) == 4:
            # Data corresponds to [FL,FR,RL,RR]
            filtered_values = [data[i] for i, leg in enumerate(self.legs) if leg in selected_legs]
        elif len(data) == 3:
            # Data corresponds to [X,Y,Z]
            filtered_values = [data[i] for i, axis in enumerate(self.axis) if axis in selected_legs]

        self.plots[name].send_data(filtered_values)

    # TODO: Make all this programmatically for any observable name
    def torque_update(self, torques, LegsAttr=False):  # noqa: D102
        self.predefine_update('Torque', torques, self.torque_legs, self.torque_joint_names, legs_attr=LegsAttr)

    def jointpos_update(self, jp, LegsAttr=False):  # noqa: D102
        self.predefine_update('JointPos', jp, self.jp_legs, self.jp_joint_names, legs_attr=LegsAttr)

    def jointvel_update(self, jv, LegsAttr=False):  # noqa: D102
        self.predefine_update('JointVel', jv, self.jv_legs, self.jv_joint_names, legs_attr=LegsAttr)

    def contact_update(self, contacts, LegsAttr=False):  # noqa: D102
        self.predefine_update('FootContacts', contacts, self.contact_legs, [], legs_attr=LegsAttr)

    def lin_acc_update(self, lin_acc):  # noqa: D102
        self.predefine_update('LinAcc', lin_acc, self.lin_acc, [], legs_attr=False)

    def ang_vel_update(self, ang_vel):  # noqa: D102
        self.predefine_update('AngVel', ang_vel, self.ang_vel, [], legs_attr=False)

    def update_plot(self):
        """Update all plots."""
        for plot in self.plots.values():
            plot.update_plot()

    def start(self):
        """Start all plots and queues."""
        for plot in self.plots.values():
            plot.start()

    def stop(self):
        """Stop all plots and queues."""
        for plot in self.plots.values():
            plot.stop()

    def reset(self):
        """Reset all plots and queues."""
        for plot in self.plots.values():
            plot.reset_queues()


# ===========================================================================
class MultiLivePlotter(mp.Process):
    """TODO: doctring."""

    def __init__(
        self,
        figure_name,
        num_subplots=2,
        window_size=50,
        subplot_titles=None,
        x_limits=None,
        y_limits=None,
        nrows=1,
        ncols=None,
        y_margin=0.1,
        plot_per_ax=1,
    ):
        """A live plotter that can handle multiple subplots, each with its own sliding window.

        :param num_subplots: Number of subplots (data streams) to display.
        :param window_size: Maximum number of data points to show in the sliding window.
        :param subplot_titles: List of titles for each subplot.
        :param x_limits: Tuple or list of tuples for x-axis limits.
        :param y_limits: Tuple or list of tuples for y-axis limits.
        :param nrows: Number of rows in the subplot layout.
        :param ncols: Number of columns in the subplot layout (default auto-calculated).
        """
        super(MultiLivePlotter, self).__init__()

        if plot_per_ax > 1 and nrows == 1 and ncols == 1:
            self.num_subplots = 1
            self.nBuffers = plot_per_ax
        else:
            self.num_subplots = nrows * ncols
            self.nBuffers = self.num_subplots
        self.window_size = window_size

        self.queue = mp.Queue()  # Queue to receive data from simulator process
        self.running = mp.Event()  # Control flag to stop the process safely

        # Each subplot gets its own deque for data storage

        self.data_buffers = [deque(maxlen=self.window_size) for _ in range(self.nBuffers)]

        self.nrows = nrows
        self.ncols = ncols
        self.y_margin = y_margin
        self.y_max = 0
        self.y_min = 100

        self.subplot_titles = subplot_titles
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.fig_name = figure_name

    def signal_handler(self, signum, frame):
        """Handles external termination signals (SIGTERM, SIGINT)."""
        print(f'[{os.getpid()}] Received signal {signum}. Shutting down gracefully...')
        self.running.clear()  # Stop the update loop
        plt.close('all')  # Close figure
        with contextlib.suppress(Exception):
            self.terminate()
        plt.close('all')  # Close figure

    def run(self):
        """This method runs in a separate process and continuously updates the plots."""
        self.running.set()  # Indicate that the plotter is running

        # Register signal handlers for clean termination
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

        # Initialize plot in a separate process
        # plt.ion()
        self.fig, axs = plt.subplots(self.nrows, self.ncols, figsize=(10, 6))
        self.axs = axs.flatten() if isinstance(axs, (list, np.ndarray)) else [axs]

        # Set figure title
        # self.fig.suptitle(self.fig_name)  # Display figure title
        self.fig.canvas.manager.set_window_title(self.fig_name)  # Set window title

        # Handle subplot titles
        if self.subplot_titles is None:
            self.subplot_titles = [f'Data {i + 1}' for i in range(self.num_subplots)]
        elif len(self.subplot_titles) < self.num_subplots:
            self.subplot_titles += [f'Data {i + 1}' for i in range(len(self.subplot_titles), self.num_subplots)]

        # Create line objects for each subplot
        self.data_buffers = [deque(maxlen=self.window_size) for _ in range(self.num_subplots)]
        self.lines = []

        for i in range(self.num_subplots):
            (line,) = self.axs[i].plot([], [], label=self.subplot_titles[i])
            self.lines.append(line)

            self.axs[i].set_title(self.subplot_titles[i])

            # Apply user-defined x and y limits
            if self.x_limits and i < len(self.x_limits):
                self.axs[i].set_xlim(*self.x_limits[i])

            if self.y_limits and i < len(self.y_limits):
                self.axs[i].set_ylim(*self.y_limits[i])
            else:
                self.axs[i].autoscale()

            self.axs[i].legend(loc='upper left')

        # Hide unused subplots
        for i in range(self.num_subplots, len(self.axs)):
            self.axs[i].axis('off')

        # Use animation for updating the plots smoothly
        self.anim = FuncAnimation(self.fig, self._update_animation, interval=10, blit=True, cache_frame_data=False)
        plt.show(block=True)  # Block execution here to keep the figure open

    def _update_animation(self, frame):
        """Function to update the animation."""
        if not self.queue.empty():
            new_values = self.queue.get()
            self._update_data(new_values)
        return self._update_plot()

    def update_data(self, new_values):
        """Update the data buffers with new values.

        Args:
            new_values:  A list of new data points for each subplot.
        """
        if self.num_subplots > 1:
            assert len(new_values) == self.num_subplots, f'Expected {self.num_subplots} values, got {len(new_values)}.'

        for i, val in enumerate(new_values):
            self.data_buffers[i].append(val)

    def _update_plot(self):
        """Refresh the plots with the updated sliding window data."""
        updated_lines = []

        for i in range(self.nBuffers):
            x_data = np.arange(len(self.data_buffers[i]))
            y_data = list(self.data_buffers[i])

            self.lines[i].set_data(x_data, y_data)
            updated_lines.append(self.lines[i])

            # Dynamically adjust y-limits
            # if  len(y_data) > 0:
            #     if(max(y_data) > self.y_max): self.y_max = max(y_data)
            #     if(min(y_data) < self.y_min): self.y_min = min(y_data)
            #     y_range = self.y_max - self.y_min

            # #   Add margin buffer
            #     margin = y_range * self.y_margin if y_range > 0 else 1
            #     self.axs[i].set_ylim(self.y_min - margin, self.y_max + margin)
            # Dynamically adjust y-limits
            # if len(y_data) > 0:
            #     y_min, y_max = min(y_data), max(y_data)
            #     y_range = y_max - y_min

            #     # Add margin buffer
            #     margin = y_range * self.y_margin if y_range > 0 else 1
            #     self.axs[i].set_ylim(y_min - margin, y_max + margin)

        return updated_lines

    def send_data(self, new_values):
        """Send new data to the plotter process.

        :param new_values: A list of new data points.
        """
        if self.running.is_set():
            if not isinstance(new_values, list):
                new_values = [new_values]

            try:
                if self.queue.qsize() >= self.window_size:
                    # Maintain sliding window effect by removing the oldest item
                    _ = self.queue.get_nowait()
                self.queue.put_nowait(new_values)  # Non-blocking
            except Exception:
                pass  # Ignore errors to avoid crashing

    def stop(self):
        """Stop the plotting process."""
        self.reset_queues()
        self.running.clear()
        os.kill(os.getpid(), signal.SIGTERM)  # Send termination signal to itself
        # self.join()  # Wait for the process to end

    def reset_queues(self):
        """Reset all stored data in the queues (data buffers)."""
        try:
            for i in range(self.num_subplots):
                self.data_buffers[i].clear()  # Clear all data
            while not self.queue.empty():  # Flush queue
                self.queue.get_nowait()
            print('Queues reset successfully.')
        except Exception:
            pass


# ===========================================================================
if __name__ == '__main__':
    # Example usage: 2 subplots, window size of 50
    # Titles, X-limits, and Y-limits for each subplot

    titles = ['Random Stream A']  # , 'Random Stream B']
    x_lims = [(0, 50), (0, 50)]
    y_lims = [(0, 2), (0, 1)]  # Different Y ranges for demonstration

    plotter = MujocoPlotter(enable=True)
    plotter.create(
        figure_name='example',
        subplot_titles=titles,
        rows=1,
        cols=2,
        y_limits=y_lims,
        window_size=50,
    )
    plotter.start()

    # Simulate data streaming for 200 updates
    while True:
        # Generate random data for each subplot
        new_val_subplot1 = random.uniform(0, 2)
        new_val_subplot2 = random.random()

        # Update the sliding windows
        plotter.plots['example'].send_data([new_val_subplot1, new_val_subplot2])

    plotter.stop()

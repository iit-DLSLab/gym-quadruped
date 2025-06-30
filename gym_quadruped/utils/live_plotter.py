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
		x_limits: list = None,	
		rows: int = 1,
		cols: int = 1,
		window_size: int = 50,
		plots_per_ax:int=1

	):
		"""
		Create new Plot figure
		"""
		if y_limits is None: y_limits = [-1, 1]
		if x_limits is None: x_limits = [(0, window_size)]	
		plotter = MultiLivePlotter(
			figure_name=figure_name,
			num_subplots=rows * cols,
			subplot_titles=subplot_titles,
			nrows=rows,
			ncols=cols,
			window_size=window_size,
			x_limits=x_limits, #[(0, window_size)] * (rows * cols),
			y_limits=y_limits * (rows * cols),
			plot_per_ax=plots_per_ax

		)
		self.plots[figure_name] = plotter

	# ---------------------------------------------------
	def predefined_plot(
		self,
		name: str,
		y_limit: list,
		legs: list = None,
		joint_names: list = None,
		window_size: int = 50,
		plot_per_ax: int = 1
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
				tmp = [f'{name} {leg}_{joint_name}_{k}' for k in range(plot_per_ax)]
				titles.append(tmp)
		self.create(
			figure_name=name,
			rows=rows,
			cols=cols,
			window_size=window_size,
			subplot_titles=titles,
			y_limits=y_limit,
			plots_per_ax=plot_per_ax,
		)
		return legs, joint_names

	def torque_plot(
		self,
		legs: list = None,
		joint_names: list = None,
		window_size: int = 50,
		enable: bool = True,
		plot_per_ax: int = 1
	):
		self.torque_enable = enable
		if enable is False or self.all_plot_enable is False: return
		self.torque_legs, self.torque_joint_names = self.predefined_plot(
			name='Torque',
			y_limit=[(-120, 120)],
			legs=legs,
			joint_names=joint_names,
			window_size=window_size,
			plot_per_ax=plot_per_ax
		)

	def jointpos_plot(
		self,
		legs: list = None,
		joint_names: list = None,
		window_size: int = 50,
		enable: bool = True,
		plot_per_ax: int = 1
	):
		self.jp_enable = enable
		if enable is False or self.all_plot_enable is False: return
		self.jp_legs, self.jp_joint_names = self.predefined_plot(
			name='JointPos',
			y_limit=[(-3.5, 3.5)],
			legs=legs,
			joint_names=joint_names,
			window_size=window_size,
			plot_per_ax=plot_per_ax
		)

	def jointvel_plot(
		self,
		legs: list = None,
		joint_names: list = None,
		window_size: int = 50,
		enable: bool = True,
		plot_per_ax: int = 1
	):
		self.jv_enable = enable
		if enable is False or self.all_plot_enable is False: return
		self.jv_legs, self.jv_joint_names = self.predefined_plot(
			name='JointVel',
			y_limit=[(-15, 15)],
			legs=legs,
			joint_names=joint_names,
			window_size=window_size,
			plot_per_ax=plot_per_ax
		)

	def footContact_plot(
		self,
		legs: list = None,
		joint_names: list = None,
		window_size: int = 50,
		enable: bool = True,
		plot_per_ax: int = 1
	):
		self.footContact_enable = enable
		if enable is False or self.all_plot_enable is False: return
		self.contact_legs, _ = self.predefined_plot(
			name='FootContacts',
			y_limit=[(-2, 2)],
			legs=legs,
			joint_names=[' '],
			window_size=window_size,
			plot_per_ax=plot_per_ax
		)
	
	def lin_acc_plot(
			self,
			axis: list=None,
			window_size: int = 50,
			enable: bool = True,
			plot_per_ax: int = 1):
		self.lin_acc_enable = enable
		if(enable is False or self.all_plot_enable is False): return

		if(axis is None): axis = self.axis

		_, self.lin_acc = self.predefined_plot(
			name='LinAcc',
			y_limit=[(-5, 13)],
			legs=["trunk"],
			joint_names=axis,
			window_size=window_size,
			plot_per_ax=plot_per_ax
		)

	def ang_vel_plot(
			self,
			axis: list=None,
			window_size: int = 50,
			enable: bool = True,
			plot_per_ax: int = 1):
		self.ang_vel_enable = enable
		if(enable is False or self.all_plot_enable is False): return

		if(axis is None): axis = self.axis

		_, self.ang_vel = self.predefined_plot(
			name='AngVel',
			y_limit=[(-5, 5)],
			legs=["Trunk"],
			joint_names=axis,
			window_size=window_size,
			plot_per_ax=plot_per_ax
		)

	def predefine_update(self, data, selected_legs, selected_joints, legs_attr=False):
		if legs_attr:
			data = [v for attr in vars(data).values() for v in (attr if isinstance(attr, list) else [attr])]
			data = np.concatenate(data).tolist()
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

			# if all values are boolean, convert to int
			if all(isinstance(v, bool) for v in filtered_values): filtered_values = [int(v) for v in filtered_values]


		elif len(data) == 3:
			# Data corresponds to [X,Y,Z]
			filtered_values = [data[i] for i, axis in enumerate(self.axis) if axis in selected_legs]

		return filtered_values

	def torque_update(self, torques, LegsAttr=False):
		if(self.torque_enable is False or self.all_plot_enable is False): return
		if(isinstance(LegsAttr,bool)): LegsAttr = [LegsAttr]
		if(isinstance(torques, list) is False): torques = [torques]  # Ensure torques is a list of lists
		plot_values = []
		for torques_values, legattr in zip(torques, LegsAttr):
			plot_values.append(self.predefine_update(torques_values, self.torque_legs, self.torque_joint_names, legs_attr=legattr))
		self.plots['Torque'].send_data(plot_values)

	def jointpos_update(self, jp, LegsAttr=False):
		if(self.jp_enable is False or self.all_plot_enable is False): return
		if(isinstance(LegsAttr,bool)): LegsAttr = [LegsAttr]
		if(isinstance(jp, list) is False): jp = [jp]  # Ensure jp is a list of lists
		plot_values = []
		for jp_values, legattr in zip(jp, LegsAttr):
			plot_values.append(self.predefine_update(jp_values, self.jp_legs, self.jp_joint_names, legs_attr=legattr))
		self.plots['JointPos'].send_data(plot_values)

	def jointvel_update(self, jv, LegsAttr=False):
		if(self.jv_enable is False or self.all_plot_enable is False): return
		if(isinstance(LegsAttr,bool)): LegsAttr = [LegsAttr]
		if(isinstance(jv, list) is False): jv = [jv]  # Ensure jv is a list of lists
		plot_values = []
		for jv_values, legattr in zip(jv, LegsAttr):
			plot_values.append(self.predefine_update(jv_values, self.jv_legs, self.jv_joint_names, legs_attr=legattr))
		self.plots['JointVel'].send_data(plot_values)

	def contact_update(self, contacts ,LegsAttr=False):
		if(self.footContact_enable is False or self.all_plot_enable is False): return
		if(isinstance(LegsAttr,bool)): LegsAttr = [LegsAttr]
		if(isinstance(contacts, list) is False): contacts = [contacts]  # Ensure contacts is a list of lists
		plot_values = []
		for contact_values,legattr in zip(contacts, LegsAttr):
			plot_values.append(self.predefine_update(contact_values, self.contact_legs, [], legs_attr=legattr))
		self.plots['FootContacts'].send_data(plot_values)
	
	def lin_acc_update(self, lin_acc):
		if(self.lin_acc_enable is False or self.all_plot_enable is False): return
		lin_acc = [list(lin_acc)]
		plot_values = []
		for val in lin_acc:
			plot_values.append(self.predefine_update(val, self.lin_acc, [], legs_attr=False))
		self.plots['LinAcc'].send_data(plot_values)
	
	def ang_vel_update(self, ang_vel):
		if(self.ang_vel_enable is False or self.all_plot_enable is False): return
		ang_vel = [list(ang_vel)]
		plot_values = []
		for val in ang_vel:
			plot_values.append(self.predefine_update(val, self.ang_vel, [], legs_attr=False))
		self.plots['AngVel'].send_data(plot_values)

	def update_plot(self):
		for plot in self.plots.values():
			plot.update_plot()

	def start(self):
		for plot in self.plots.values():
			plot.start()

	def stop(self):
		for plot in self.plots.values():
			plot.stop()

	def reset(self):
		for plot in self.plots.values():
			plot.reset_queues()


# ===========================================================================
class MultiLivePlotter(mp.Process):
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
		plot_per_ax=1

	):
		"""
		A live plotter that can handle multiple subplots, each with its own sliding window.

		:param num_subplots:   Number of subplots (data streams) to display.
		:param window_size:    Maximum number of data points to show in the sliding window.
		:param subplot_titles: List of titles for each subplot. 
		:param x_limits:       Tuple or list of tuples for x-axis limits.
		:param y_limits:       Tuple or list of tuples for y-axis limits.
		:param nrows:          Number of rows in the subplot layout.
		:param ncols:          Number of columns in the subplot layout (default auto-calculated).
		"""
		super(MultiLivePlotter, self).__init__()

		self.multiplots = False
		if(plot_per_ax > 1): self.multiplots = True
		self.num_subplots = nrows * ncols * plot_per_ax
		self.nBuffers = nrows * ncols
		self.window_size = window_size
		self.plots_per_ax = plot_per_ax	

		self.y_queue = mp.Queue()  # Queue to receive data from simulator process
		#self.x_queue = mp.Queue()  # Queue to receive data from simulator process
		self.running = mp.Event()  # Control flag to stop the process safely

		# Each subplot gets its own deque for data storage		
		self.data_buffers = [[deque(maxlen=self.window_size) for _ in range(self.plots_per_ax)] for _ in range(self.nBuffers)]

		
		#self.x_axis = [deque(maxlen=self.window_size) for _ in range(self.nBuffers)]

		self.nrows = nrows
		self.ncols = ncols
		self.y_margin = y_margin
		self.y_max = 0
		self.y_min = 100
		self.axis_counter = 0

		self.subplot_titles = subplot_titles
		self.x_limits = x_limits
		self.y_limits = y_limits
		self.fig_name = figure_name

	def signal_handler(self, signum, frame):
		"""
		Handles external termination signals (SIGTERM, SIGINT).
		"""
		print(f'[{os.getpid()}] Received signal {signum}. Shutting down gracefully...')
		self.running.clear()  # Stop the update loop
		plt.close('all')  # Close figure
		with contextlib.suppress(Exception):
			self.terminate()
		plt.close('all')  # Close figure

	def run(self):
		"""
		This method runs in a separate process and continuously updates the plots.
		"""
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
		self.data_buffers = [[deque(maxlen=self.window_size) for _ in range(self.plots_per_ax)] for _ in range(self.nBuffers)]
		self.lines = []

		for i in range(self.nBuffers):
			tmp = []
			for j in range(self.plots_per_ax):
				(line,) = self.axs[i].plot([], [], label=self.subplot_titles[i][j], color = np.random.rand(3,))  # Random color for each line
				tmp.append(line)

				self.axs[i].set_title(self.subplot_titles[i][0])

				# Apply user-defined x and y limits
				#if self.x_limits and i < len(self.x_limits):
				self.axs[i].set_xlim(self.x_limits[0])

				if self.y_limits and i < len(self.y_limits):
					self.axs[i].set_ylim(*self.y_limits[i])
				else:
					self.axs[i].autoscale()

				self.axs[i].legend(loc='upper left')
			self.lines.append(tmp)

		# Hide unused subplots
		for i in range(self.nBuffers, len(self.axs)):
			self.axs[i].axis('off')

		# Use animation for updating the plots smoothly
		self.anim = FuncAnimation(self.fig, self._update_animation, interval=10, blit=True, cache_frame_data=False)
		plt.show(block=True)  # Block execution here to keep the figure open

	def _update_animation(self, frame):
		"""
		Function to update the animation.
		"""
		if not self.y_queue.empty():
			new_values = self.y_queue.get()
			#x_values = self.x_queue.get()
			self._update_data(new_values)

		return self._update_plot()

	def _update_data(self, new_values):
		"""
		Update the sliding windows with new data for each subplot.
		The order of the input is important, it will direct the values to their
		specific plot
		"""

		# if(self.num_subplots>1):
		# 	assert len(new_values) == self.num_subplots, f'Expected {self.num_subplots} values, got {len(new_values)}.'

		for i in range(self.nBuffers): 
			for j in range(self.plots_per_ax): 
				print(new_values)
				print(self.nBuffers, self.plots_per_ax)
				a = new_values[i][j]
				print(a)
				self.data_buffers[i][j].append(new_values[i][j]) 
		
	
	def _update_plot(self):

		"""
		Refresh the plots with the updated sliding window data.
		"""
		updated_lines = []

		for i in range(self.nBuffers):
			for j in range(self.plots_per_ax):
				data = list(self.data_buffers[i][j])
				y_data, x_data = [], []

				for elem in data:
					if(isinstance(elem, tuple)):
						y_data.append(elem[0])
						x_data.append(elem[1])
					else:
						y_data.append(elem)
						x_data = range(len(data))
				
				self.lines[i][j].set_data(x_data, y_data)
				updated_lines.append(self.lines[i][j])

			#x_data = np.arange(self.last_update, self.last_update + len(self.data_buffers[i]))
			#x_data = list(self.x_axis[i])
			# data = list(self.data_buffers[i])
			# y_data, x_data = [], []

			# for elem in data:
			# 	if(isinstance(elem, tuple)):
			# 		y_data.append(elem[0])
			# 		x_data.append(elem[1])
			# 	else:
			# 		y_data.append(elem)
			# 		x_data = range(len(data))

			# if(self.multiplots):
			# 	#self.axs.set_xlim(min(x_data), max(x_data))
			# 	self.lines[i].set_data(x_data, y_data)
			# 	updated_lines.append(self.lines[i])
			# else:
			# 	try: self.axs[i].set_xlim(min(x_data), max(x_data))
			# 	except: pass
			# 	self.lines[i].set_data(x_data, y_data)
			# 	updated_lines.append(self.lines[i])

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

	def send_data(self, new_data):
		"""
		Accepts flexible input formats:
		- send_data([0, 1, 2])                      → [[0], [1], [2]]
		- send_data([[0, 1, 2], [3, 4, 5]])         → [[0, 3], [1, 4], [2, 5]]
		"""
		if not self.running.is_set():
			return

		if isinstance(new_data, list):
			if all(isinstance(el, list) for el in new_data):
				# Case: send_data([[0,1,2],[3,4,5]])
				if len(set(len(lst) for lst in new_data)) != 1:
					raise ValueError("All inner lists must have the same length.")
				zipped = [list(t) for t in zip(*new_data)]
			else:
				# Case: send_data([0,1,2])
				zipped = [[v] for v in new_data]
		else:
			raise ValueError("send_data expects a list or list of lists.")

		try:
			if self.y_queue.qsize() >= self.window_size:
				_ = self.y_queue.get_nowait()
			self.y_queue.put_nowait(zipped)
		except Exception:
			pass

	def stop(self):
		"""
		Stop the plotting process.
		"""
		self.reset_queues()
		self.running.clear()
		os.kill(os.getpid(), signal.SIGTERM)  # Send termination signal to itself
		# self.join()  # Wait for the process to end

	def reset_queues(self):
		"""
		Reset all stored data in the queues (data buffers).
		"""
		try:
			for i in range(self.nBuffers):
				for j in range(self.plots_per_ax):
					self.data_buffers[i][j].clear()  # Clear all data
			while not self.y_queue.empty():  # Flush queue
				self.y_queue.get_nowait()
			print('Queues reset successfully.')
		except Exception:
			pass
		self.axis_counter = 0	


# ===========================================================================
if __name__ == '__main__':
	# Example usage: 2 subplots, window size of 50
	# Titles, X-limits, and Y-limits for each subplot

	titles = [['Random Stream A', 'Random Stream B']]
	#x_lims = [(0, 50), (0, 50)]
	y_lims = [(-2, 2), (0, 1)]  # Different Y ranges for demonstration

	plotter = MujocoPlotter(enable=True)
	plotter.create(
		figure_name='example',
		subplot_titles=titles,
		rows=1,
		cols=1,
		y_limits=y_lims,
		x_limits=[(0, 50)],
		window_size=50,
		plots_per_ax=2
	)

	plotter.start()

	# Simulate data streaming for 200 updates
	while True:
		# Generate random data for each subplot
		new_val_subplot1 = [random.uniform(0, 0.5)]
		new_val_subplot2 = [random.uniform(0,0.5)]

		# Update the sliding windows
		plotter.plots['example'].send_data(new_data = [new_val_subplot1, new_val_subplot2])

	plotter.stop()
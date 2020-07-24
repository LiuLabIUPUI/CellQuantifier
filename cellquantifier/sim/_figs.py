import matplotlib.pyplot as plt
import numpy as np
from ..plot import format_ax


def show_traj(ax, traj_df, ndim, color='red'):

	"""Plots a single trajectory in 2D or 3D

	Parameters
	----------

	ax: axis object
		Matplotlib axis object to use for plotting

	traj_df : DataFrame
		Contains 'x', 'y', 'particle' columns
	"""

	if ndim == 2:
			ax.plot(traj_df['x'], \
					traj_df['y'], \
					linewidth=1,
					color=color)
	if ndim == 3:
			ax.plot(traj_df['x'], \
					traj_df['y'], \
					traj_df['z'], \
					linewidth=1,
					color=color)

def show_traj_batch(traj_df, origins=None, ndim=3):

	"""Plots the trajectories in 2D or 3D

	Parameters
	----------
	origins : 1D ndarray
		Contains the origin of each particle e.g. lattice, ring, etc.
		that can be used to show the origins of the particles

	"""

	nparticles = traj_df['particle'].nunique()

	fig = plt.figure()
	if ndim == 3:
		ax = fig.gca(projection='3d')
	else:
		ax = fig.gca()

	colors = plt.cm.get_cmap('viridis')
	colors = colors(np.linspace(0, 1, nparticles))
	for i in range(nparticles):
		this_df = traj_df.loc[traj_df['particle'] == i]
		show_traj(ax, this_df, ndim, color=colors[i])

	tst = traj_df.loc[traj_df['frame'] == 0]
	if origins is not None:
		ax.scatter(tst['x'],tst['y'], c='r', s=20)
	#
	# format_ax(ax,
	# 		  xlabel=r'$\mathbf{x}$',
	# 		  ylabel=r'$\mathbf{y}$',
	# 		  show_legend=False,
	# 		  label_fontsize=15)
	plt.show()

def show_force_comp(traj_df):

	"""Show x,y,z components of the stochastic force

	Parameters
	----------

	"""

	# """
	# ~~~~~~~Initialize the figure~~~~~~~~~~
	# """

	nparticles = traj_df['particle'].nunique()
	fig,ax = plt.subplots(1,3, figsize=(15,5))
	colors = plt.cm.get_cmap('coolwarm')
	colors = colors(np.linspace(0, 1, nparticles))


	# """
	# ~~~~~~~Plot the components~~~~~~~~~
	# """

	for n in range(nparticles):
		this_df = traj_df.loc[traj_df['particle'] == n]

		a_x = this_df['a_x'].to_numpy()
		a_y = this_df['a_y'].to_numpy()
		a_z = this_df['a_z'].to_numpy()

		ax[0].plot(a_x*1e6, color=colors[n], alpha=1)
		ax[1].plot(a_y*1e6, color=colors[n], alpha=1)
		ax[2].plot(a_z*1e6, color=colors[n], alpha=1)

	labels = [r'$\mathbf{\xi_{x}/m(\frac{\mu m}{s^{2}})}$',\
			  r'$\mathbf{\xi_{y}/m(\frac{\mu m}{s^{2}})}$',\
			  r'$\mathbf{\xi_{z}/m(\frac{\mu m}{s^{2}})}$']

	for i, x in enumerate(ax):
		x = format_ax(x,
					  xlabel=r'$\mathbf{Time}$',
					  ylabel=labels[i],
					  label_fontsize=15,
					  show_legend=False,
					  ax_is_box=False)
	plt.tight_layout()
	plt.show()

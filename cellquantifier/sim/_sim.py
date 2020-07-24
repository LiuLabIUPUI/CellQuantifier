from numpy.random import normal, uniform
import numpy as np
import pandas as pd

def get_trajectory_1(nframes, dcoeff, origin=None):

	"""
	Build a trajectory where each displacement is selected from a gaussian
	distribution N(m, sigma) where sigma is constant. Displacements are measured
	relative to preceding position

	Use this method when msd = <r(t+n) - r(t)> (individual msd)

	Parameters
	----------
	"""

	columns = ['frame', 'particle', 'x', 'y','dx','dy']
	traj_df = pd.DataFrame(columns=columns)
	traj_df['frame'] = np.arange(0, nframes, 1)
	traj_df['sigma'] = np.sqrt(2*dcoeff)
	traj_df['dx'] = normal(size=nframes, scale=traj_df['sigma'])
	traj_df['dy'] = normal(size=nframes, scale=traj_df['sigma'])

	if origin is None:
		x0,y0 = (0,0)
	else:
		x0, y0 = origin

	traj_df['x'] = traj_df['dx'].cumsum() + x0
	traj_df['y'] = traj_df['dy'].cumsum() + y0

	return traj_df

def get_trajectory_2(nframes, dcoeff, origin=None):

	"""
	Build a trajectory where each displacement is selected from a gaussian
	distribution N(m, sigma) where sigma is a function of time. Displacements
	are measured relative to the origin

	Use this method when msd = <r(t) - r(0)> (ensemble msd)

	Parameters
	----------
	"""

	columns = ['frame', 'particle', 'x', 'y','dx','dy']
	traj_df = pd.DataFrame(columns=columns)
	traj_df['frame'] = np.arange(0, nframes, 1)
	traj_df['sigma'] = np.sqrt(2*dcoeff*traj_df['frame'])
	traj_df['dx'] = normal(size=nframes, scale=traj_df['sigma'])
	traj_df['dy'] = normal(size=nframes, scale=traj_df['sigma'])

	if origin is None:
		x0,y0 = (0,0)
	else:
		x0, y0 = origin

	traj_df['x'] = traj_df['dx'] + x0
	traj_df['y'] = traj_df['dy'] + y0

	return traj_df

def get_trajectories(nparticles,
					 nframes,
					 dcoeff=10,
					 method='individual',
					 origins=None):

	"""Build a set of trajectories based on gaussian distributed displacements

	Parameters
	----------
	framerate: float,
			   frames per second
	"""

	columns = ['frame', 'particle', 'x', 'y','dx','dy']
	df = pd.DataFrame(columns=columns)

	if origins is None:
		origins = [None for n in range(nparticles)]

	for n in range(nparticles):
		if method == 'individual':
			this_df = get_trajectory_1(nframes, dcoeff, origin=origins[n])
		elif method == 'ensemble':
			this_df = get_trajectory_2(nframes, dcoeff, origin=origins[n])

		this_df = this_df.assign(particle=n)
		df = pd.concat([df, this_df])

	return df

import numpy as np
import pandas as pd
import trackpy as tp
import cellquantifier.math as cq_math

def track_blobs(blobs_df,
			    search_range=3,
				memory=5,
				min_traj_length=10,
				filters=None):

	"""
	Wrapper for trackpy library functions (assign detection instances to particle trajectories)

	Parameters
	----------
	blobs_df : DataFrame
		DataFrame with column for frame number and x,y particle coordinates

	filters: dict
		dictionary of filters

	Returns
	-------
	blobs_df_tracked : DataFrame object
		DataFrame with added column for particle number

	Examples
	--------
	>>> from cellquantifier import data
	>>> from cellquantifier.smt.detect import detect_blobs
	>>> blobs_df = detect_blobs(data.simulated_cell(), threshold=.05)
	>>> blobs_df, im = track(blobs_df)
	>>> blobs_df

			 x      y  sigma_raw
	0    286.0  361.0        7.5
	1    286.0  292.0        4.5
	2    286.0  246.0        3.0
	3    286.0   46.0        4.5
	4    285.0  349.0        6.0
	..     ...    ...        ...
	620   20.0  349.0        3.0
	621   19.0  336.0        9.0
	622   18.0  346.0        3.0
	623   18.0  343.0        3.0
	624   18.0  331.0        4.5

		"""


	# """
	# ~~~~~~~~~~~Link Trajectories and Filter Stubs~~~~~~~~~~~~~~
	# """

	blobs_df = tp.link_df(blobs_df, search_range=search_range, memory=memory)
	blobs_df = tp.filter_stubs(blobs_df, min_traj_length)
	blobs_df = blobs_df.reset_index(drop=True)

	# """
	# ~~~~~~~~~~~Get Secondary Parameters~~~~~~~~~~~~~~
	# """

	blobs_df = blobs_df.sort_values(['particle', 'frame'])

	blobs_df['mass'] = 2*np.pi*blobs_df['sigma_x']*blobs_df['sigma_y']*blobs_df['amp']
	blobs_df['area'] = np.pi*blobs_df['sigma_x']*blobs_df['sigma_y']
	blobs_df['dist_err'] = np.linalg.norm(blobs_df[['x', 'y']].values.astype(np.float64)  - blobs_df[['x_raw', 'y_raw']].values.astype(np.float64), axis=1)
	blobs_df['delta_area'] = np.abs((blobs_df.groupby('particle')['area'].apply(pd.Series.pct_change)))
	blobs_df['sigx_to_sigraw'] = blobs_df['sigma_x']/blobs_df['sigma_raw']
	blobs_df['sigy_to_sigraw'] = blobs_df['sigma_y']/blobs_df['sigma_raw']

	# """
	# ~~~~~~~~~~~Get Individual Particle D Values~~~~~~~~~~~~~~
	# """

	blobs_df = get_d_values(blobs_df, im, config)
	blobs_df = blobs_df.apply(pd.to_numeric)

	# """
	# ~~~~~~~~~~~Filter DataFrame and Relink~~~~~~~~~~~~~~
	# """

	if filters:

		blobs_df = filter_df(blobs_df, filters)
		blobs_df = link(blobs_df)

	im = tp.imsd(blobs_df, resolution, frame_rate)

	return blobs_df, im

def get_d_values(traj_df, im, config):

	"""Returns a modififed traj_df with an extra column for each particles diffusion coefficient"""

	n = int(round(len(im.index)/config.DIVIDE_NUM))
	im = im.head(n)

	#get diffusion coefficient of each particle
	particles = np.unique(traj_df['particle'])
	for particle in particles:

		msd = im[particle].to_numpy()
		msd = msd*1e6 #convert to nm
		popt = cq_math.fit_msd_log(im.index.values, msd)

		traj_df.loc[traj_df['particle'] == particle, 'D'] = popt[0]
		traj_df.loc[traj_df['particle'] == particle, 'alpha'] = popt[1]

	return traj_df

def filter_df(blobs_df, filters):

	blobs_df = blobs_df[blobs_df.dist_err < filters['max_dist_err']]
	blobs_df = blobs_df[blobs_df.delta_area < filters['max_delta_area']]
	blobs_df = blobs_df[blobs_df.sigx_to_sigraw < filters['sig_to_sigraw']]
	blobs_df = blobs_df[blobs_df.sigy_to_sigraw < filters['sig_to_sigraw']]

	return traj_df

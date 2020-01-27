from cellquantifier.plot.plotutil import add_t_test, add_strip_plot
import matplotlib.pyplot as plt
import sys
import re
import pandas as pd
import numpy as np

def add_heat_map(ax,
				 df,
				 mean_col,
				 bin_col,
				 cat_col,
				 xlabel=None,
				 ylabel=None,
				 nbins=8,
				 inter_pts=100,
				 hole_size = 10,
				 edge_ring=False,
				 pixel_size=.1083):


	"""Add heat maps to the axis

	Parameters
	----------
	ax: object
		matplotlib axis to annotate

	df: DataFrame

	mean_col: str
		Column color coded by the heat map e.g. diffusion coefficient

	bin_col: str
		Column to use to bin the data e.g. avg_dist_bound

	cat_col: str
		Column to use for categorical sorting

	nbins: int
		Number of bins to create from bin_col, used for interolation

	inter_pts: int
		Number of data points after interolation

	"""

	# """
	# ~~~~~~~~~~~Bin data~~~~~~~~~~~~~~
	# """

	df['category'] = pd.cut(df[bin_col], nbins, labels=False)

	# """
	# ~~~~~~~~~~~Separate data by category~~~~~~~~~~~~~~
	# """

	cats = df[cat_col].unique()
	dfs = [df.loc[df[cat_col] == cat] for cat in cats]

	# """
	# ~~~~~~~~~~~Compute means, std errors~~~~~~~~~~~~~~
	# """

	grouped_dfs = [df.groupby('category') for df in dfs]
	means = [df[mean_col].mean().to_numpy() for df in grouped_dfs]
	stds = [np.divide(df[mean_col].std().to_numpy(),\
			np.sqrt(df[mean_col].count().to_numpy())) for df in grouped_dfs]

	# """
	# ~~~~~~~~~~~Get radial domain~~~~~~~~~~~~~~
	# """

	#ensure the radial domain is the same for every category
	r_min = df[bin_col].to_numpy().min()
	r_max = df[bin_col].to_numpy().max()
	bin_size = (r_max-r_min)/nbins
	hist, r = np.histogram(df[bin_col], nbins) #get bin edges

	# """
	# ~~~~~~~~~Ensure data was found in all bins for all categories~~~~~~~~~~~~~~
	# """

	x = lambda means: all([len(mean) == nbins for mean in means])
	if not x(means):
		print("One or more bins are empty!")
		sys.exit()

	# """
	# ~~~~~~~~~~~Interpolate the data~~~~~~~~~~~~~~
	# """

	r_discrete = 0.5*(r[1:] + r[:-1]) #get bin centers
	min_center, max_center = r_discrete[0], r_discrete[-1]
	r_cont = np.linspace(min_center, max_center, inter_pts)

	inter = [np.interp(r_cont, r_discrete, mean) for mean in means]
	min = np.concatenate(inter).min()
	max = np.concatenate(inter).max()

	# """
	# ~~~~~~~~~~~Pad r_cont and interpolated data~~~~~~~~~~~~~~
	# """

	pad_size = 20
	r_pad = np.linspace(min_center-hole_size, min_center, pad_size)
	inter_pad = np.full(pad_size, 0)
	r_cont = np.concatenate((r_pad, r_cont), axis=0)
	inter = [np.concatenate((inter_pad, _inter), axis=0) for _inter in inter]

	# """
	# ~~~~~~~~~~~Find where to put the second ring (53bp1 boundary)~~~~~~~~~~~~~~
	# """

	idx = np.abs(r_cont).argmin()

	# """
	# ~~~~~~~~~~~Generate polar plot~~~~~~~~~~~~~~
	# """

	theta = np.linspace(0,2*np.pi,100)
	r, theta = np.meshgrid(r_cont, theta)
	tmp = (r ** 2.0) / 4.0 #placeholder function

	r_arr = [np.zeros_like(tmp) for i in range(len(means))]

	for i, r_i in enumerate(r_arr):
		for j in range(r_arr[0].shape[0]):
			inter[i][inter[i] == 0] = None
			if edge_ring:
				inter[i][idx:idx+3] = None
			r_i[j,:] = inter[i]

		ax[i].set_yticks(r_discrete)
		ax[i].set_xticklabels([])
		ax[i].set_yticklabels([])
		cb = ax[i].pcolormesh(theta, r, r_i, cmap='coolwarm', vmin=min, vmax=max)
		cb = plt.colorbar(cb, ax=ax[i])
		cb.set_label(ylabel)
		ax[i].grid(True, axis='y', color='black', linewidth=1)

		# ax_labels = [re.sub("[^0-9]", "", item.get_text()) \
		# 		  for item in ax[i].get_yticklabels()]
		# ax[i].set_yticklabels(ax_labels)

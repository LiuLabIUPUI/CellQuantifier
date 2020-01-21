from cellquantifier.plot.plotutil import add_t_test, add_strip_plot
from scipy.stats import binned_statistic
from skimage.draw import circle_perimeter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

def add_heat_map(ax,
				 df,
				 mean_col,
				 bin_col,
				 cat_col,
				 xlabel=None,
				 ylabel=None,
				 nbins=8,
				 interp_pts=100,
				 width=0.45):


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
		Number of bins to create from bin_col, used for interpolation

	interp_pts: int
		Number of data points after interpolation

	Example
	----------
	from cellquantifier.plot.plotutil import add_heat_map
	import pandas as pd
	import matplotlib.pyplot as plt

	fig = plt.figure(figsize=(10,5))

	ax1 = fig.add_subplot(121, projection='polar')
	ax2 = fig.add_subplot(122, projection='polar')
	ax = [ax1,ax2]

	df = pd.read_csv('/home/cwseitz/Desktop/physDataMerged.csv')
	df = df.drop_duplicates('particle')

	xlabel = r'$\mathbf{D (nm^{2}/s)}$'
	ylabel = r'$\mathbf{D (nm^{2}/s)}$'

	add_heat_map(ax,
	             df,
	             'D',
	             'avg_dist_bound',
	             'exp_label',
	              xlabel=xlabel,
	              ylabel=ylabel)

	plt.tight_layout()
	plt.show()

	"""

	# """
	# ~~~~~~~~~~~Get data~~~~~~~~~~~~~~
	# """

	df['category'] = pd.cut(df[bin_col], nbins)
	cats = df[cat_col].unique()
	dfs = [df.loc[df[cat_col] == cat] for cat in cats]
	grouped_dfs = [df.groupby('category') for df in dfs]

	means = [df[mean_col].mean().to_numpy() for df in grouped_dfs]

	stds = [np.divide(df[mean_col].std().to_numpy(),\
			np.sqrt(df[mean_col].count().to_numpy())) for df in grouped_dfs]

	radial_bins = df['category'].unique()
	radial_bins = np.flip(sorted([round(np.abs(x.right)) for x in radial_bins]))
	radial_bins = np.round(radial_bins*108*1e-3, 1)

	# """
	# ~~~~~~~~~~~Interpolate the data~~~~~~~~~~~~~~
	# """

	x = np.linspace(0, len(means[0]), interp_pts)
	xp = np.linspace(0, len(means[0]), nbins)

	interp = [np.interp(x, xp, mean) for mean in means]

	theta = np.linspace(0,2*np.pi,100)
	x, th = np.meshgrid(x, theta)
	tmp = (x ** 2.0) / 4.0

	r_all = [np.zeros_like(tmp) for i in range(len(means))]

	for i, r_x in enumerate(r_all):
		for j in range(r_all[0].shape[0]):
			r_x[j,:] = interp[i]

	min = np.concatenate(interp).min()
	max = np.concatenate(interp).max()

	for i, r_x in enumerate(r_all):
		cb = ax[i].pcolormesh(th, x, r_x, cmap='coolwarm', vmin=min, vmax=max)
		cb = plt.colorbar(cb, ax=ax[i])
		cb.set_label(ylabel)

		ax[i].set_xticklabels([])
		ax[i].set_yticklabels([])

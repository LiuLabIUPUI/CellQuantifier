import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import trackpy as tp
import pandas as pd

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
from ..math import msd, fit_msd
from skimage.io import imsave
from ..plot.plotutil import plt2array


def plot_msd_merged(blobs_df,
			 cat_col,
			 pixel_size=.1084,
			 frame_rate=3.3,
			 divide_num=5):


	fig, ax = plt.subplots(1,3)
	cats = blobs_df[cat_col].unique()
	blobs_dfs = [blobs_df.loc[blobs_df[cat_col] == cat] for cat in cats]
	colors = plt.cm.jet(np.linspace(0,1,len(cats)))

	#
	# ~~~~~~~~~~~Check if blobs_df is empty~~~~~~~~~~~~~~
	# """

	for i, blobs_df in enumerate(blobs_dfs):

		if blobs_df.empty:
			return

		# Calculate individual msd
		im = tp.imsd(blobs_df, mpp=pixel_size, fps=frame_rate)

		#Get the diffusion coefficient for each individual particle
		D_ind = blobs_df.drop_duplicates('particle')['D'].mean()

		# """
		# ~~~~~~~~~~~Get the avg/stdev for D,alpha of each particle~~~~~~~~~~~~~~
		# """

		if len(im) > 1:

			d_avg = (blobs_df.drop_duplicates(subset='particle')['D']).mean() #average D value
			d_std = (blobs_df.drop_duplicates(subset='particle')['D']).std() #stdev of D value

			#avg/stdev of alpha
			alpha_avg = (blobs_df.drop_duplicates(subset='particle')['alpha']).mean() #average alpha value
			alpha_std = (blobs_df.drop_duplicates(subset='particle')['alpha']).std() #stdev of alpha value

	# """
	# ~~~~~~~~~~~Get the mean MSD curve and its standard dev~~~~~~~~~~~~~~
	# """

		#cut the msd curves and convert units to nm
		n = int(round(len(im.index)/divide_num))
		im = im.head(n)
		im = im*1e6

		imsd_mean = im.mean(axis=1)
		imsd_std = im.std(axis=1, ddof=0)

		x = imsd_mean.index
		y = imsd_mean.to_numpy()
		yerr = imsd_std.to_numpy()

		t = imsd_mean.index.to_numpy()
		popt_log = fit_msd(t,y, space='log') #fit the average msd curve in log space
		popt_lin = fit_msd(t,y, space='linear') #fit the average msd curve in linear space

	# """
	# ~~~~~~~~~~~Plot the fit of the average and the average of fits~~~~~~~~~~~~~~
	# """

		fit_of_avg = msd(t, popt_log[0], popt_log[1])
		avg_of_fits = msd(t, d_avg, alpha_avg)

		ax[0].errorbar(x, avg_of_fits, yerr=yerr, linestyle='None',
				marker='o', color=colors[i])

		ax[0].plot(t, avg_of_fits, '-', color=colors[i],
				   linewidth=4, markersize=12, label="Average of Fit")


		ax[0].set_xlabel(r'$\tau (\mathbf{s})$')
		ax[0].set_ylabel(r'$\langle \Delta r^2 \rangle$ [$nm^2$]')
		ax[0].legend()

		# """
		# ~~~~~~~~~~~Add D value histogram~~~~~~~~~~~~~~
		# """

		ax[1].hist(blobs_df.drop_duplicates(subset='particle')['D'].to_numpy(),
					bins=30, color=colors[i], label=cats[i])

		ax[1].legend(loc='upper right')

		ax[1].set_ylabel('Frequency')
		ax[1].set_xlabel('D')

		# """
		# ~~~~~~~~~~~Add alpha value histogram~~~~~~~~~~~~~~
		# """

		ax[2].hist(blobs_df.drop_duplicates(subset='particle')['alpha'].to_numpy(),
					bins=30, color=colors[i], label=cats[i])

		ax[2].legend(loc='upper right')

		ax[2].set_ylabel('Frequency')
		ax[2].set_xlabel('alpha')

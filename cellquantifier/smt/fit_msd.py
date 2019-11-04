import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
from ..math import msd, fit_msd

mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'

def plot_msd(im,
			 blobs_df,
			 image,
			 output_path,
			 root_name,
			 resolution=.1084,
			 divide_num=5):

	"""
	Generates the 3 panel msd figure with color-coded trajectories, msd curves, and a histogram of d values

	Parameters
	----------

	im: DataFrame:
		Mean squared displacement, each column is a particle

	blobs_df : DataFrame
		DataFrame containing 'D', 'frame', 'x', and 'y' columns

	image: 2D ndarray
		The image the trajectories will be plotted on

	output_path: string
		The folder where output files should be saved

	root_name: string
		The root file name to use for output files

	resolution: float
		The resolution of the images in microns/pixel

	divide_num: int
		The number used to divide the msd curves


	Returns
	-------
	blobs_df_tracked : DataFrame object
		DataFrame with added column for particle number

	Examples
	--------
	>>> from cellquantifier import data
	>>> from cellquantifier.smt.detect import detect_blobs, detect_blobs_batch
	>>> from cellquantifier.smt.fit_psf import fit_psf, fit_psf_batch
	>>> from cellquantifier.smt.track import track_blobs
	>>> from cellquantifier.smt.fit_msd import plot_msd
	>>> frames = data.simulated_cell()
	>>> blobs_df, det_plt_array = detect_blobs_batch(frames)
	>>> psf_df, fit_plt_array = fit_psf_batch(frames, blobs_df)
	>>> blobs_df, im = track_blobs(psf_df, min_traj_length=10)
	>>> d, alpha = plot_msd(im,
						 blobs_df,
						 image=frames[0],
						 output_path='/home/cwseitz/Desktop/',
						 root_name='file',
						 resolution=.1084,
						 divide_num=5)
		"""

	#Get the diffusion coefficient for each individual particle
	D_ind = blobs_df.drop_duplicates('particle')['D'].mean()

	#Plot the image
	fig, ax = plt.subplots(1, 3, figsize=(20, 8))
	ax[0].imshow(image, cmap='gray', aspect='equal')

	# """
	# ~~~~~~~~~~~Add D value scale bar to left plot~~~~~~~~~~~~~~
	# """

	scalebar = ScaleBar(resolution, 'um', location = 'upper right')
	ax[0].add_artist(scalebar)

	divider = make_axes_locatable(ax[0])
	ax_cb = divider.new_horizontal(size="5%", pad=0.1)
	norm = mpl.colors.Normalize(vmin = blobs_df['D'].min(), vmax = blobs_df['D'].max())
	cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=mpl.cm.jet, norm=norm, orientation='vertical')
	cb1.set_label(r'$\mathbf{D (nm^{2}/s)}$')
	fig.add_axes(ax_cb)

	colormap = plt.cm.get_cmap('jet')
	blobs_df['D_norm'] = blobs_df['D']/(blobs_df['D'].max()) #normalize D column to maximum D value

	# """
	# ~~~~~~~~~~~Plot the color coded trajectories~~~~~~~~~~~~~~
	# """

	particles = blobs_df.particle.unique()
	for i in range(len(particles)):
		traj = blobs_df[blobs_df.particle == particles[i]]
		traj = traj.sort_values(by='frame')
		ax[0].plot(traj.y, traj.x, color=colormap(traj['D_norm'].mean()))

	ax[0].set_aspect(1.0)
	ax[0].set(xlabel='y (pixel)', ylabel='x (pixel)')


	# """
	# ~~~~~~~~~~~Plot the MSD curves for each particle~~~~~~~~~~~~~~
	# """

	#cut the msd curves and convert units to nm
	n = int(round(len(im.index)/divide_num))
	im = im.head(n)
	im = im*1e6

	if len(im) > 1:
		ax[1].plot(im, 'k-', alpha=0.3)

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
	# ~~~~~~~~~~~Save D and alpha values per particle~~~~~~~~~~~~~~
	# """

		blobs_df.drop_duplicates(subset='particle')[['D', 'alpha']].to_csv(output_path + root_name + "-Dalpha.csv", index=False)


	# """
	# ~~~~~~~~~~~Get the mean MSD curve and its standard dev~~~~~~~~~~~~~~
	# """

		imsd_mean = im.mean(axis=1)
		imsd_std = im.std(axis=1, ddof=0)
		imsd_mean.to_csv(output_path + root_name + "-msdData.csv", header=True)

		x = imsd_mean.index
		y = imsd_mean.to_numpy()
		yerr = imsd_std.to_numpy()

		ax[1].errorbar(x, y, yerr=yerr, linestyle='None',
				marker='o', color='black')

		t = imsd_mean.index.to_numpy()
		popt_log = fit_msd(t,y, space='log') #fit the average msd curve in log space
		popt_lin = fit_msd(t,y, space='linear') #fit the average msd curve in linear space

	# """
	# ~~~~~~~~~~~Plot the fit of the average and the average of fits~~~~~~~~~~~~~~
	# """

		fit_of_avg = msd(t, popt_log[0], popt_log[1])
		avg_of_fits = msd(t, d_avg, alpha_avg)

		ax[1].plot(t, fit_of_avg, '-', color='b',
				   linewidth=4, markersize=12, label="Fit of Average")
		ax[1].plot(t, avg_of_fits, '-', color='r',
				   linewidth=4, markersize=12, label="Average of Fit")
	# """
	# ~~~~~~~~~~~Write out the averages and errors to text box~~~~~~~~~~~~~~
	# """

		textstr = '\n'.join((

			r'$D_{FOA}: %.2f \pm %.2f \mathbf{nm^{2}/s}$' % (popt_log[0], d_std),
			r'$\alpha_{FOA}: %.2f \pm %.2f$' % (popt_log[1], alpha_std),

			r'$D_{AOF}: %.2f \pm %.2f \mathbf{nm^{2}/s}$' % (d_avg, d_std),
			r'$\alpha_{AOF}: %.2f \pm %.2f$' % (alpha_avg, alpha_std)))


		props = dict(boxstyle='round', facecolor='wheat', alpha=0.0)
		ax[1].text(.1, .8, textstr, transform=ax[1].transAxes,  horizontalalignment='left', verticalalignment='top', fontsize=14, color='black', bbox=props)


	ax[1].set_xlabel(r'$\tau (\mathbf{s})$')
	ax[1].set_ylabel(r'$\langle \Delta r^2 \rangle$ [$nm^2$]')
	ax[1].legend()

	# """
	# ~~~~~~~~~~~Add D value histogram~~~~~~~~~~~~~~
	# """

	ax[2].hist(blobs_df.drop_duplicates(subset='particle')['D'].to_numpy(), bins=10, color='gray')
	ax[2].set_ylabel('Frequency')
	ax[2].set_xlabel(r'$D (\mathbf{nm^{2}/s})$')

	plt.tight_layout()
	plt.savefig(output_path + root_name + "-msdPlot.png")

	return d_avg, alpha_avg

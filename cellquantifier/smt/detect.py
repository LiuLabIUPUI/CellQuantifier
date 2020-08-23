import numpy as np; import pandas as pd; import pims
from ..plot.plotutil import anno_blob, anno_scatter
from ..plot.plotutil import plot_end
from matplotlib_scalebar.scalebar import ScaleBar
from skimage.feature import blob_log
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from ..deno import median
from ..io import imshow_gray
from skimage.util import img_as_ubyte
from scipy.ndimage import gaussian_laplace
from skimage.filters.thresholding import _cross_entropy
from skimage.util import img_as_float
from skimage.feature import peak_local_max
from skimage.filters import gaussian

def detect_blobs(pims_frame,

				min_sig=1,
				max_sig=3,
				num_sig=5,
				blob_thres=0.1,
				peak_thres_rel=0.1,

				r_to_sigraw=3,
				pixel_size = .1084,

				diagnostic=True,
				pltshow=True,
				plot_r=True,
				blob_marker='^',
				blob_markersize=10,
				blob_markercolor=(0,0,1,0.8),
				truth_df=None):
	"""
	Detect blobs for each frame.

	Parameters
	----------
	pims_frame : pims.Frame object
		Each frame in the format of pims.Frame.
	min_sig : float, optional
		As 'min_sigma' argument for blob_log().
	max_sig : float, optional
		As 'max_sigma' argument for blob_log().
	num_sig : int, optional
		As 'num_sigma' argument for blob_log().
	blob_thres : float, optional
		As 'threshold' argument for blob_log().
	peak_thres_rel : float, optional
		Relative peak threshold [0,1].
		Blobs below this relative value are removed.
	r_to_sigraw : float, optional
		Multiplier to sigraw to decide the fitting patch radius.
	pixel_size : float, optional
		Pixel size in um. Used for the scale bar.
	diagnostic : bool, optional
		If true, run the diagnostic.
	pltshow : bool, optional
		If true, show diagnostic plot.
	plot_r : bool, optional
		If True, plot the blob boundary.
	truth_df : DataFrame or None. optional
		If provided, plot the ground truth position of the blob.

	Returns
	-------
	blobs_df : DataFrame
		columns = ['frame', 'x', 'y', 'sig_raw', 'r',
					'peak', 'mass', 'mean', 'std']
	plt_array :  ndarray
		ndarray of diagnostic plot.

	Examples
	--------
	import pims
	from cellquantifier.smt.detect import detect_blobs, detect_blobs_batch
	frames = pims.open('cellquantifier/data/simulated_cell.tif')
	detect_blobs(frames[0])
	"""

	# """
	# ~~~~~~~~~~~~~~~~~Detection using skimage.feature.blob_log~~~~~~~~~~~~~~~~~
	# """

	frame = pims_frame

	# threshold automation
	if blob_thres=='auto':
		frame_f = img_as_float(frame)
		frame_log = -gaussian_laplace(frame_f, sigma=min_sig) * min_sig**2

		maxima = peak_local_max(frame_log,
					threshold_abs=0,
					footprint=None,
					num_peaks=10)

		columns = ['x', 'y', 'peak_log',]
		maxima_df = pd.DataFrame([], columns=columns)
		maxima_df['x'] = maxima[:, 0]
		maxima_df['y'] = maxima[:, 1]
		maxima_df['peak_log'] = frame_log[ maxima_df['x'], maxima_df['y'] ]

		blob_thres_final = maxima_df['peak_log'].mean()*0.005
	else:
		blob_thres_final = blob_thres

	# # peak_thres_rel automation
	# if peak_thres_rel=='auto':
	# 	maxima = peak_local_max(frame,
	# 				threshold_abs=0,
	# 				footprint=None,
	# 				num_peaks=10)
	#
	# 	columns = ['x', 'y', 'peak',]
	# 	maxima_df = pd.DataFrame([], columns=columns)
	# 	maxima_df['x'] = maxima[:, 0]
	# 	maxima_df['y'] = maxima[:, 1]
	# 	maxima_df['peak'] = frame[ maxima_df['x'], maxima_df['y'] ]
	#
	# 	peak_thres_abs = maxima_df['peak'].mean()*0.05

	blobs = blob_log(frame,
					 min_sigma=min_sig,
					 max_sigma=max_sig,
					 num_sigma=num_sig,
					 threshold=blob_thres_final)

	# """
	# ~~~~~~~~~~~~~~~~~~~~~~Prepare blobs_df and update it~~~~~~~~~~~~~~~~~~~~~~
	# """

	columns = ['frame', 'x', 'y', 'sig_raw', 'r', 'peak', 'mass']
	blobs_df = pd.DataFrame([], columns=columns)
	blobs_df['x'] = blobs[:, 0]
	blobs_df['y'] = blobs[:, 1]
	blobs_df['sig_raw'] = blobs[:, 2]
	blobs_df['r'] = blobs[:, 2] * r_to_sigraw
	blobs_df['frame'] = pims_frame.frame_no
	# """
	# ~~~~~~~Filter detections at the edge~~~~~~~
	# """
	blobs_df = blobs_df[(blobs_df['x'] - blobs_df['r'] > 0) &
				  (blobs_df['x'] + blobs_df['r'] + 1 < frame.shape[0]) &
				  (blobs_df['y'] - blobs_df['r'] > 0) &
				  (blobs_df['y'] + blobs_df['r'] + 1 < frame.shape[1])]
	for i in blobs_df.index:
		x = int(blobs_df.at[i, 'x'])
		y = int(blobs_df.at[i, 'y'])
		r = int(round(blobs_df.at[i, 'r']))
		blob = frame[x-r:x+r+1, y-r:y+r+1]
		blobs_df.at[i, 'peak'] = blob.max()
		blobs_df.at[i, 'mass'] = blob.sum()

	# """
	# ~~~~~~~Filter detections~~~~~~~
	# """
	if peak_thres_rel=='auto' and blob_thres!='auto':
		blobs_df_nofilter = blobs_df.copy()

		maxima = peak_local_max(frame,
					threshold_abs=0,
					footprint=None,
					num_peaks=10)

		columns = ['x', 'y', 'peak',]
		maxima_df = pd.DataFrame([], columns=columns)
		maxima_df['x'] = maxima[:, 0]
		maxima_df['y'] = maxima[:, 1]
		maxima_df['peak'] = frame[ maxima_df['x'], maxima_df['y'] ]

		peak_thres_abs = maxima_df['peak'].mean()*0.05
		blobs_df = blobs_df[(blobs_df['peak'] > peak_thres_abs)]
	elif peak_thres_rel=='auto' and blob_thres=='auto':
		# blobs_df['peak_norm'] = (blobs_df['peak']-blobs_df['peak'].min()) \
		# 				/ (blobs_df['peak'].max()-blobs_df['peak'].min()) * 10
		# blobs_df['r_norm'] = (blobs_df['r']-blobs_df['r'].min()) \
		# 				/ (blobs_df['r'].max()-blobs_df['r'].min())
		# blobs_df['mass_norm'] = (blobs_df['mass']-blobs_df['mass'].min()) \
		# 				/ (blobs_df['mass'].max()-blobs_df['mass'].min()) * 10
		#
		# blobs_df['peak_times_r'] = blobs_df['peak_norm'] * blobs_df['r_norm']
		# blobs_df = blobs_df.sort_values(by='peak_times_r', ascending=False)
		# pk_by_r_thres = blobs_df.head(10)['peak_times_r'].mean()*0.1
		#
		# blobs_df['mass_times_r'] = blobs_df['mass_norm'] * blobs_df['r_norm']
		# blobs_df = blobs_df.sort_values(by='mass_times_r', ascending=False)
		# mass_by_r_thres = blobs_df.head(10)['mass_times_r'].mean()*0
		#
		# blobs_df = blobs_df.sort_values(by='peak_norm', ascending=False)
		# peak_thres_abs = blobs_df.head(10)['peak_norm'].mean()*0
		#
		# blobs_df = blobs_df.sort_values(by='mass_norm', ascending=False)
		# mass_thres_abs = blobs_df.head(10)['mass_norm'].mean()*0
		#
		# blobs_df_nofilter = blobs_df.copy()
		# blobs_df = blobs_df[ (blobs_df['peak_times_r']>=pk_by_r_thres) ]
		# blobs_df = blobs_df[ (blobs_df['mass_times_r']>=mass_by_r_thres) ]
		# blobs_df = blobs_df[ (blobs_df['peak_norm'] >= peak_thres_abs) ]
		# blobs_df = blobs_df[ (blobs_df['mass_norm'] >= mass_thres_abs) ]


		blobs_df['peak_times_r'] = blobs_df['peak'] * blobs_df['r']
		blobs_df = blobs_df.sort_values(by='peak_times_r', ascending=False)
		pk_by_r_thres = blobs_df.head(10)['peak_times_r'].mean()*0.15

		blobs_df['mass_times_r'] = blobs_df['mass'] * blobs_df['r']
		blobs_df = blobs_df.sort_values(by='mass_times_r', ascending=False)
		mass_by_r_thres = blobs_df.head(10)['mass_times_r'].mean()*0.3

		blobs_df = blobs_df.sort_values(by='peak', ascending=False)
		peak_thres_abs = blobs_df.head(10)['peak'].mean()*0.7

		blobs_df = blobs_df.sort_values(by='mass', ascending=False)
		mass_thres_abs = blobs_df.head(10)['mass'].mean()*0.25

		blobs_df_nofilter = blobs_df.copy()
		blobs_df = blobs_df[ (blobs_df['peak_times_r']>pk_by_r_thres) ]
		blobs_df = blobs_df[ (blobs_df['mass_times_r']>mass_by_r_thres) ]
		blobs_df = blobs_df[ (blobs_df['peak'] > peak_thres_abs) ]
		blobs_df = blobs_df[ (blobs_df['mass'] > mass_thres_abs) ]
	else:
		blobs_df_nofilter = blobs_df.copy()

		peak_thres_abs = blobs_df['peak'].max() * peak_thres_rel
		blobs_df = blobs_df[(blobs_df['peak'] > peak_thres_abs)]

	# """
	# ~~~~~~~~~~~~~~~~~~~~~~~~~Print detection summary~~~~~~~~~~~~~~~~~~~~~~~~~
	# """

	if len(blobs_df)==0:
		print("\n"*3)
		print("##############################################")
		print("ERROR: No blobs detected in this frame!!!")
		print("##############################################")
		print("\n"*3)
		return pd.DataFrame(np.array([])), np.array([])
	else:
		print("Det in frame %d: %s" % (pims_frame.frame_no, len(blobs_df)))

	# """
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Diagnostic~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# """

	plt_array = []
	if diagnostic:
		fig, ax = plt.subplots(2, 2, figsize=(12,12))

		# """
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~Annotate the blobs~~~~~~~~~~~~~~~~~~~~~~~~~~
		# """
		ax[0][0].imshow(frame, cmap="gray", aspect='equal')
		anno_blob(ax[0][0], blobs_df_nofilter, marker=blob_marker, markersize=blob_markersize,
				plot_r=plot_r, color=blob_markercolor)
		ax[0][0].text(0.95,
				0.05,
				"Foci_num: %d" %(len(blobs_df_nofilter)),
				horizontalalignment='right',
				verticalalignment='bottom',
				fontsize = 12,
				color = (0.5, 0.5, 0.5, 0.5),
				transform=ax[0][0].transAxes,
				weight = 'bold',
				)

		ax[0][1].imshow(frame, cmap="gray", aspect='equal')
		anno_blob(ax[0][1], blobs_df, marker=blob_marker, markersize=blob_markersize,
				plot_r=plot_r, color=blob_markercolor)
		ax[0][1].text(0.95,
				0.05,
				"Foci_num: %d" %(len(blobs_df)),
				horizontalalignment='right',
				verticalalignment='bottom',
				fontsize = 12,
				color = (0.5, 0.5, 0.5, 0.5),
				transform=ax[0][1].transAxes,
				weight = 'bold',
				)

		# """
		# ~~~~~~~~~~~~~~~~~~~Annotate ground truth if needed~~~~~~~~~~~~~~~~~~~
		# """
		if isinstance(truth_df, pd.DataFrame):
			anno_scatter(ax[0][0], truth_df, marker='o', color=(0,1,0,0.8))

		# """
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~Add scale bar~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# """
		font = {'family': 'arial', 'weight': 'bold','size': 16}
		scalebar = ScaleBar(pixel_size, 'um', location = 'upper right',
			font_properties=font, box_color = 'black', color='white')
		scalebar.length_fraction = .3
		scalebar.height_fraction = .025
		ax[0][0].add_artist(scalebar)

		# """
		# ~~~~Plot foci in parameter space~~~~
		# """
		# x2, y2 = blobs_df_nofilter['r_norm'], blobs_df_nofilter['peak_norm']
		# ax[1][0].scatter(x2, y2, marker='^', c=[(0,0,1)])
		#
		# x3, y3 = blobs_df_nofilter['r_norm'], blobs_df_nofilter['mass_norm']
		# ax[1][1].scatter(x3, y3, marker='^', c=[(0,0,1)])
		#
		# if peak_thres_rel=='auto' and blob_thres=='auto':
		# 	x2_thres = np.linspace(0.05, 1, 50)
		# 	y2_thres = pk_by_r_thres / x2_thres
		# 	y2_peak = x2_thres / x2_thres * peak_thres_abs
		# 	ax[1][0].plot(x2_thres, y2_thres, '--', c=(0,0,0,0.8), linewidth=3)
		# 	ax[1][0].plot(x2_thres, y2_peak, '--', c=(0,0,0,0.8), linewidth=3)
		#
		# 	x3_thres = np.linspace(0.05, 1, 50)
		# 	y3_thres = mass_by_r_thres / x3_thres
		# 	y3_peak = x3_thres / x3_thres * mass_thres_abs
		# 	ax[1][1].plot(x3_thres, y3_thres, '--', c=(0,0,0,0.8), linewidth=3)
		# 	ax[1][1].plot(x3_thres, y3_peak, '--', c=(0,0,0,0.8), linewidth=3)


		x2, y2 = blobs_df_nofilter['r'], blobs_df_nofilter['peak']
		ax[1][0].scatter(x2, y2, marker='^', c=[(0,0,1)])

		x3, y3 = blobs_df_nofilter['r'], blobs_df_nofilter['mass']
		ax[1][1].scatter(x3, y3, marker='^', c=[(0,0,1)])

		if peak_thres_rel=='auto' and blob_thres=='auto':
			x2_thres = np.linspace(min_sig, max_sig, 50)
			y2_thres = pk_by_r_thres / x2_thres
			y2_peak = x2_thres / x2_thres * peak_thres_abs
			ax[1][0].plot(x2_thres, y2_thres, '--', c=(0,0,0,0.8), linewidth=3)
			ax[1][0].plot(x2_thres, y2_peak, '--', c=(0,0,0,0.8), linewidth=3)

			x3_thres = np.linspace(min_sig, max_sig, 50)
			y3_thres = mass_by_r_thres / x3_thres
			y3_peak = x3_thres / x3_thres * mass_thres_abs
			ax[1][1].plot(x3_thres, y3_thres, '--', c=(0,0,0,0.8), linewidth=3)
			ax[1][1].plot(x3_thres, y3_peak, '--', c=(0,0,0,0.8), linewidth=3)

		plt_array = plot_end(fig, pltshow)

	return blobs_df, plt_array


def detect_blobs_batch(pims_frames,
			min_sig=1,
			max_sig=3,
			num_sig=5,
			blob_thres=0.1,
			peak_thres_rel=0.1,
			r_to_sigraw=3,
			pixel_size = 108.4,
			diagnostic=False,
			pltshow=False,
			plot_r=True,
			blob_marker='^',
			blob_markersize=10,
			blob_markercolor=(0,0,1,0.8),
			truth_df=None):

	"""
	Detect blobs for the whole movie.

	Parameters
	----------
	See detect_blobs().

	Returns
	-------
	blobs_df : DataFrame
		columns = ['frame', 'x', 'y', 'sig_raw', 'r',
					'peak', 'mass', 'mean', 'std']
	plt_array :  ndarray
		ndarray of diagnostic plots.

	Examples
	--------
	import pims
	from cellquantifier.smt.detect import detect_blobs, detect_blobs_batch
	frames = pims.open('cellquantifier/data/simulated_cell.tif')
	detect_blobs_batch(frames, diagnostic=0)
	"""

	# """
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Prepare blobs_df~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# """

	columns = ['frame', 'x', 'y', 'sig_raw', 'r', 'peak', 'mass', 'mean', 'std']
	blobs_df = pd.DataFrame([], columns=columns)
	plt_array = []

	# """
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Update blobs_df~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# """

	for i in range(len(pims_frames)):
		current_frame = pims_frames[i]
		fnum = current_frame.frame_no
		if isinstance(truth_df, pd.DataFrame):
			current_truth_df = truth_df[truth_df['frame'] == fnum]
		else:
			current_truth_df = None

		tmp, tmp_plt_array = detect_blobs(pims_frames[i],
					   min_sig=min_sig,
					   max_sig=max_sig,
					   num_sig=num_sig,
					   blob_thres=blob_thres,
					   peak_thres_rel=peak_thres_rel,
					   r_to_sigraw=r_to_sigraw,
					   pixel_size=pixel_size,
					   diagnostic=diagnostic,
					   pltshow=pltshow,
					   plot_r=plot_r,
					   blob_marker=blob_marker,
					   blob_markersize=blob_markersize,
					   blob_markercolor=blob_markercolor,
					   truth_df=current_truth_df)
		blobs_df = pd.concat([blobs_df, tmp], sort=True)
		plt_array.append(tmp_plt_array)

	blobs_df.index = range(len(blobs_df))
	plt_array = np.array(plt_array)

	return blobs_df, plt_array

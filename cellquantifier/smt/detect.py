import numpy as np; import pandas as pd; import pims
from cellquantifier.util.plot_util.anno import anno_blob, anno_scatter
from cellquantifier.util.plot_util._plot_end import plot_end
from matplotlib_scalebar.scalebar import ScaleBar
from skimage.feature import blob_log
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def detect_blobs(pims_frame,
				min_sig=1,
				max_sig=3,
				num_sig=5,
				blob_thres=0.1,
				peak_thres_rel=0.1,
				r_to_sigraw=3,
				pixel_size = 108.4,
				diagnostic=True,
				pltshow=True,
				plot_r=True,
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
		Pixel size in nm. Used for the scale bar.
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
	blobs = blob_log(frame,
	                 min_sigma=min_sig,
	                 max_sigma=max_sig,
	                 num_sigma=num_sig,
	                 threshold=blob_thres)

	# """
    # ~~~~~~~~~~~~~~~~~~~~~~Prepare blobs_df and update it~~~~~~~~~~~~~~~~~~~~~~
    # """

	columns = ['frame', 'x', 'y', 'sig_raw', 'r', 'peak', 'mass', 'mean', 'std']
	blobs_df = pd.DataFrame([], columns=columns)
	blobs_df['x'] = blobs[:, 0]
	blobs_df['y'] = blobs[:, 1]
	blobs_df['sig_raw'] = blobs[:, 2]
	blobs_df['r'] = blobs[:, 2] * r_to_sigraw
	blobs_df['frame'] = pims_frame.frame_no
	for i in blobs_df.index:
	    x = int(blobs_df.at[i, 'x'])
	    y = int(blobs_df.at[i, 'y'])
	    r = int(round(blobs_df.at[i, 'r']))
	    blob = frame[x-r:x+r+1, y-r:y+r+1]
	    blobs_df.at[i, 'mass'] = blob.sum()
	    blobs_df.at[i, 'std'] = blob.std()
	    blobs_df.at[i, 'peak'] = blob.max()
	    blobs_df.at[i, 'mean'] = blob.mean()

	# """
    # ~~~~~~~Filter detections at the edge and those below peak_thres_abs~~~~~~~
    # """

	peak_thres_abs = blobs_df['peak'].max() * peak_thres_rel
	blobs_df = blobs_df[(blobs_df['x'] - blobs_df['r'] > 0) &
				  (blobs_df['x'] + blobs_df['r'] + 1 < frame.shape[0]) &
				  (blobs_df['y'] - blobs_df['r'] > 0) &
				  (blobs_df['y'] + blobs_df['r'] + 1 < frame.shape[1]) &
				  (blobs_df['peak'] > peak_thres_abs)]

	# """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~Print detection summary~~~~~~~~~~~~~~~~~~~~~~~~~
    # """

	if len(blobs_df)==0:
		print('\n'*3+'#' * 50 \
                +'\nERROR: No blobs detected in this frame!!!\n' \
                +'#' * 50+'\n'*3)
		return pd.DataFrame(np.array([])), np.array([])
	else:
		print("Det in frame %d: %s" % (pims_frame.frame_no, len(blobs_df)))

	# """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Diagnostic~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # """

	plt_array = []
	if diagnostic:
	    fig = plt.figure(figsize=(12,9))
	    ax0 = plt.subplot2grid((6,9),(0,0), colspan=6, rowspan=6)

		# """
	    # ~~~~~~~~~~~~~~~~~~~~~~~~~~Annotate the blobs~~~~~~~~~~~~~~~~~~~~~~~~~~
	    # """
	    ax0.imshow(frame, cmap="gray", aspect='equal')
	    anno_blob(ax0, blobs_df, marker='^',
	            plot_r=plot_r, color=(0,0,1,0.8))

		# """
	    # ~~~~~~~~~~~~~~~~~~~Annotate ground truth if needed~~~~~~~~~~~~~~~~~~~
	    # """
	    if isinstance(truth_df, pd.DataFrame):
	        anno_scatter(ax0, truth_df, marker='o', color=(0,1,0,0.8))

		# """
	    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~Add scale bar~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	    # """
	    font = {'family': 'arial', 'weight': 'bold','size': 16}
	    scalebar = ScaleBar(pixel_size, 'nm', location = 'upper right',
	        font_properties=font, box_color = 'black', color='white')
	    scalebar.length_fraction = .3
	    scalebar.height_fraction = .025
	    ax0.add_artist(scalebar)

		# """
	    # ~~~~~~~~~~~~~~~~~Add histograms of several properties~~~~~~~~~~~~~~~~~
	    # """
	    ax1 = plt.subplot2grid((6,9),(1,6), colspan=3)
	    ax1.hist(blobs_df['mass']/blobs_df['mass'].max(),
	            bins=20, density=1, color=(0,0,0,0.3))
	    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	    ax1.set(xlabel='relative blob intensity mass', ylabel='weight (a.u)')
	    ax1.set_xlim([0,1])
	    ax2 = plt.subplot2grid((6,9),(2,6), colspan=3)
	    ax2.hist(blobs_df['mean']/blobs_df['mean'].max(), bins=20, density=1,
	               color=(0,0,0,0.3))
	    ax2.set(xlabel='relative blob intensity mean', ylabel='weight (a.u)')
	    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	    ax2.set_xlim([0,1])
	    ax3 = plt.subplot2grid((6,9),(3,6), colspan=3)
	    ax3.hist(blobs_df['std']/blobs_df['std'].max(), bins=20, density=1,
	               color=(0,0,0,0.3))
	    ax3.set(xlabel='relative blob intensity std', ylabel='weight (a.u)')
	    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	    ax3.set_xlim([0,1])
	    ax4 = plt.subplot2grid((6,9),(4,6), colspan=3)
	    ax4.hist(blobs_df['peak']/blobs_df['peak'].max(), bins=20, density=1,
	               color=(0,0,0,0.3))
	    ax4.set(xlabel='relative blob intensity peak', ylabel='weight (a.u)')
	    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	    ax4.set_xlim([0,1])

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
                       truth_df=current_truth_df)
        blobs_df = pd.concat([blobs_df, tmp])
        plt_array.append(tmp_plt_array)

    blobs_df.index = range(len(blobs_df))
    plt_array = np.array(plt_array)

    return blobs_df, plt_array

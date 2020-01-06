import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from scipy.stats import pearsonr
from cellquantifier.math import fit_linear


def add_det_scatter(ax, blobs_df, plot_r=False):

	"""Plot pie chart
	Parameters
	----------

    ax : object
        matplotlib axis to annotate ellipse.

	labels: list, optional
		list of labels for each slice of the pie chart

	nbins: int, optional
		number of slices to use for pie chart

	kwargs: dict
		dictionary where keys are plot labels and values are array-like

	"""

	# """
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~Annotate the blobs~~~~~~~~~~~~~~~~~~~~~~~~~~
	# """

	ax.imshow(frame, cmap="gray", aspect='equal')
	anno_blob(ax, blobs_df, marker='^',
			plot_r=plot_r, color=(0,0,1,0.8))

	# """
	# ~~~~~~~~~~~~~~~~~~~Annotate ground truth if needed~~~~~~~~~~~~~~~~~~~
	# """
	if isinstance(truth_df, pd.DataFrame):
		anno_scatter(ax, truth_df, marker='o', color=(0,1,0,0.8))

	# """
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~Add scale bar~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# """
	font = {'family': 'arial', 'weight': 'bold','size': 16}
	scalebar = ScaleBar(pixel_size, 'um', location = 'upper right',
		font_properties=font, box_color = 'black', color='white')
	scalebar.length_fraction = .3
	scalebar.height_fraction = .025
	ax.add_artist(scalebar)

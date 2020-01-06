import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import pearsonr
from cellquantifier.math import fit_linear

def add_scatter(ax,
				df,
				col1,
				col2,
				col1_alias=None,
				col2_alias=None,
				norm=False,
				color='blue'):

	"""Generate scatter plot of two variables

	Parameters
	----------

	ax : object
		matplotlib axis object

	"""

	x,y = df[col1], df[col2]

	if norm:
		x = x/np.abs(x).max()
		y = y/np.abs(y).max()

	ax.scatter(x, y, c=color, s=10)

	# """
	# ~~~~~~~~~~~Set the label~~~~~~~~~~~~~~
	# """

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['left'].set_linewidth(2)
	ax.spines['bottom'].set_linewidth(2)
	ax.tick_params(labelsize=13, width=2, length=5)

	if col1_alias != None and col2_alias != None:

		ax.set_xlabel(col1_alias, fontsize=15)
		ax.set_ylabel(col2_alias, fontsize=15)

	else:

		ax.set_xlabel(col1, fontsize=15)
		ax.set_ylabel(col2, fontsize=15)

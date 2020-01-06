import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import pearsonr
from cellquantifier.math import fit_linear

def add_corr_coeff(ax, df, col1, col2, norm=False):

	"""Add pearson correlation coefficient to axis

	Parameters
	----------

	ax : object
		matplotlib axis object

	"""

	x,y = df[col1], df[col2]

	if norm:
		x = x/np.abs(x).max()
		y = y/np.abs(y).max()

	slope, intercept, r, p = fit_linear(x,y)

	ax.text(0.9,
			0.9,
			r'$\mathbf{\rho}$' + """: % .2f""" %(r),
			fontsize = 12,
			color = 'black',
			transform=ax.transAxes)

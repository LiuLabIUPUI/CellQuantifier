import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import pearsonr
from cellquantifier.math import fit_linear

def add_scatter(ax, df, col1, col2, norm=False, fit=False, color='blue'):

	"""Generate scatter plot of two variables typically for correlation use

	Parameters
	----------

	ax : object
		matplotlib axis to annotate ellipse.

	"""

	x,y = df[col1], df[col2]

	if norm:
		x = x/x.max()
		y = y/y.max()

	slope, intercept, r, p = fit_linear(x,y)

	if fit:
		ax.plot(x, intercept + slope*x, 'lime', label='fitted line')

	ax.scatter(x, y, label=pearsonr, c=color, s=10)
	ax.text(0.75,
			0.75,
			r'$\mathbf{\rho}$' + """: % .2f""" %(r),
			fontsize = 12,
			color = 'black',
			transform=ax.transAxes)

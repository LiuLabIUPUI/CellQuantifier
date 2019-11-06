import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_hist(labels=None, **kwargs):

	"""Plot histogram

	Parameters
	----------

	kwargs: dict
		dictionary where keys are plot labels and values are array-like

	labels: list, optional
		list of labels for x,y axes

	Example
	-------
	>>>from cellquantifier.plot import plot_hist
	>>>path1 = 'cellquantifier/data/test_Dalpha.csv'
	>>>path2 = 'cellquantifier/data/test_Dalpha2.csv'
	>>>df1 = pd.read_csv(path1, index_col=None, header=0)
	>>>df2 = pd.read_csv(path2, index_col=None, header=0)
	>>>labels = [r'D (nm$^2$/s)','Weight (a.u)']
	>>>plot_hist(labels, damaged=df1['D'], control=df2['D'])

	"""

	from cellquantifier.util.stats import t_test

	fig, ax = plt.subplots()
	colors = plt.cm.jet(np.linspace(0,1,len(kwargs)))
	x = []

	for key, value in kwargs.items():

		ax.hist(value, bins=30, color=colors[list(kwargs.keys()).index(key)], density=True, label=key)
		ax.legend(loc='upper right')

		if type(value) is not np.ndarray:
			value = value.to_numpy()
		x.append(value)

	if labels:
		ax.set(xlabel=labels[0], ylabel=labels[1])

	if len(kwargs) == 2:
		t,p = t_test(x[0], x[1])
		ax.text(0.75,
				0.75,
				"""t-value: %.2f""" %(t),
				fontsize = 12,
				color = 'black',
				transform=ax.transAxes)

	plt.show()

def scatter_bivariate(x,y, labels=None, fit=False):

	"""Generate scatter plot of two variables typically for correlation use

	Parameters
	----------

	x,y: (N,) array_like

	labels: list, optional
		list of labels for x,y axes

	fit: bool, optional
		whether or not to perform linear regression

	Example
	-------
	>>>from cellquantifier.plot import scatter_bivariate
	>>>path = 'cellquantifier/data/test_Dalpha.csv'
	>>>df = pd.read_csv(path, index_col=None, header=0)
	>>>x, y = df['D'], df['alpha']
	>>>scatter_bivariate(x,y, labels=['D','alpha'], fit=True)


	"""
	from scipy.stats import pearsonr
	from cellquantifier.math import fit_linear

	fig, ax = plt.subplots()

	slope, intercept, r, p = fit_linear(x,y)

	if fit:
		ax.plot(x, intercept + slope*x, 'lime', label='fitted line')

	if labels:
		ax.set(xlabel=labels[0], ylabel=labels[1])

	ax.scatter(x, y, label=pearsonr, color='orange', s=10)
	ax.text(0.75,
			0.75,
			r'$\mathbf{\rho}$' + """: % .2f""" %(r),
			fontsize = 12,
			color = 'blue',
			transform=ax.transAxes)


	plt.show()

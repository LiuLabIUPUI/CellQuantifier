import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_hist(labels, **kwargs):

	"""Plot histogram

	Parameters
	----------

	labels: list of labels for x and y axes

	kwargs: dictionary where keys are plot labels and values are array-like

	Example
	-------
	>>>from cellquantifier.plot import plot_d_hist
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
		ax.set(xlabel=labels[0], ylabel=labels[1])

		if type(value) is not np.ndarray:
			value = value.to_numpy()
		x.append(value)

	if len(kwargs) == 2:
		t,p = t_test(x[0], x[1])
		ax.text(0.75,
				0.75,
				"""t-value: %.2f""" %(t),
				fontsize = 12,
				color = 'black',
				transform=ax.transAxes)

	plt.show()

def scatter_bivariate(path,x,y, fit=False):

	"""Generate scatter plot of two variables typically for correlation use

	Parameters
	----------

	path: the file path to the csv containing the fittData dataframe

	x,y: (N,) array_like

	fit: bool
		whether or not to perform linear regression

	Example
	-------
	>>>from cellquantifier.plot import scatter_bivariate
	>>>path = 'cellquantifier/data/test_Dalpha.csv'
	>>>scatter_bivariate(path, 'D', 'alpha', fit=True)

	"""
	from scipy.stats import pearsonr
	from cellquantifier.math import fit_linear

	fig, ax = plt.subplots()
	df = pd.read_csv(path, index_col=None, header=0)

	slope, intercept, r, p = fit_linear(df[x],df[y])

	if fit:
		ax.plot(df[x], intercept + slope*df[x], 'lime', label='fitted line')

	ax.scatter(df[x], df[y], label=pearsonr, color='orange', s=10)
	ax.set(xlabel=x, ylabel=y)
	ax.text(0.75,
			0.75,
			r'$\mathbf{\rho}$' + """: % .2f""" %(r),
			fontsize = 12,
			color = 'blue',
			transform=ax.transAxes)


	plt.show()

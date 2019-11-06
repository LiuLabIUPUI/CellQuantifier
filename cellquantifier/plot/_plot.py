import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_hist(labels=None, nbins=10, **kwargs):

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

		ax.hist(value, bins=nbins, color=colors[list(kwargs.keys()).index(key)], density=True, label=key, alpha=.75)
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
	plt.tight_layout()
	plt.show()

def plot_cc_hist(data, nbins=10, labels=None):

	"""Plot histogram

	Parameters
	----------

	data: DataFrame
		two column DataFrame where the first column will be used for binning, second for color coding

	nbins: int, optional
		number of bins to use for histogram

	labels: list, optional
		labels for x,y axes

	Example
	-------
	>>>from cellquantifier.plot import plot_hist
	>>>path = 'cellquantifier/data/test_fittData.csv'
	>>>labels = ['Distance to COM', 'Weight (a.u.)']
	>>>df = pd.read_csv(path, index_col=None, header=0)
	>>>plot_cc_hist(df[['dist_to_com', 'D']], nbins=30, labels=labels)

	"""

	import matplotlib as mpl
	from matplotlib import colors
	from mpl_toolkits.axes_grid1 import make_axes_locatable

	fig, ax = plt.subplots()
	data['bin'] = pd.cut(data.iloc[:,0], nbins)

	size = data.groupby(['bin']).size()
	average = data.groupby(['bin']).mean().iloc[:,1]
	N, bins, patches = ax.hist(data.iloc[:,0], bins=nbins, normed=True)

	fracs =  average / average.max()
	norm = colors.Normalize(fracs.min(), fracs.max())
	for thisfrac, thispatch in zip(fracs, patches):
	    color = plt.cm.jet(norm(thisfrac))
	    thispatch.set_facecolor(color)

	divider = make_axes_locatable(ax)
	ax_cb = divider.new_horizontal(size="5%", pad=0.1)
	norm = mpl.colors.Normalize(vmin = average.min(), vmax = average.max())
	cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=mpl.cm.jet, norm=norm, orientation='vertical')
	cb1.set_label(r'$\mathbf{D (nm^{2}/s)}$')
	fig.add_axes(ax_cb)

	if labels:
		ax.set(xlabel=labels[0], ylabel=labels[1])

	plt.tight_layout()
	plt.show()

def plot_pie(labels=None, nbins=3, **kwargs):

	"""Plot pie chart
	Parameters
	----------

	labels: list, optional
		list of labels for each slice of the pie chart

	nbins: int, optional
		number of slices to use for pie chart

	kwargs: dict
		dictionary where keys are plot labels and values are array-like

	Example
	-------
	>>>from cellquantifier.plot import plot_pie
	>>>path1 = 'cellquantifier/data/test_Dalpha.csv'
	>>>path2 = 'cellquantifier/data/test_Dalpha2.csv'
	>>>df1 = pd.read_csv(path1, index_col=None, header=0)
	>>>df2 = pd.read_csv(path2, index_col=None, header=0)
	>>>labels = ['slow', 'medium', 'fast']
	>>>plot_pie(labels, damaged=df1['D'], control=df2['D'])
	"""

	fig, ax = plt.subplots(1, len(kwargs))

	i = 0
	for key, value in kwargs.items():

		value = value.to_numpy()
		hist, edges = np.histogram(value, normed=False, bins=3)

		ax[i].pie(hist, labels=labels, autopct='%1.1f%%',
		        shadow=True, startangle=90)
		ax[i].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
		ax[i].set_title(key)

		i+=1

	plt.tight_layout()
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

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_d_hist(*args, **kwargs):

	"""Plot histogram of D values

	Parameters
	----------

	args: list of information to place in the plot legend (one element per line)

	kwargs: dictionary where keys are plot labels and values are file paths

	Returns
	-------
	None

	Examples
	--------
	>>>from cellquantifier.plot import plot_d_hist
	>>>path1 = 'cellquantifier/data/test_d_values1.csv'
	>>>path2 = 'cellquantifier/data/test_d_values2.csv'
	>>>plot_d_hist(damaged=path1, control=path2)

	"""

	fig, ax = plt.subplots()
	colors = plt.cm.jet(np.linspace(0,1,len(kwargs)))

	for key, value in kwargs.items():

		df = pd.read_csv(value, index_col=None, header=0)
		ax.hist(df['D'], bins=30, color=colors[list(kwargs.keys()).index(key)], density=True, label=key)
		ax.legend(loc='upper right')
		ax.set(xlabel=r'D (nm$^2$/s)', ylabel='Weight (a.u)')

	for arg in args:
		ax.text(0.5,
				0.75,
				"""T-test: % 6.2f""" %(arg),
				fontsize = 12,
				color = 'black',
				transform=ax.transAxes)

	plt.show()

def plot_alpha_hist(*args, **kwargs):

	"""Plot histogram of D values

	Parameters
	----------

	args: list of information to place in the plot legend (one element per line)

	kwargs: dictionary where keys are plot labels and values are file paths

	Returns
	-------
	None

	Examples
	--------
	>>>from cellquantifier.plot import plot_alpha_hist
	>>>path1 = 'cellquantifier/data/test_Dalpha.csv'
	>>>path2 = 'cellquantifier/data/test_Dalpha2.csv'
	>>>plot_alpha_hist(damaged=path1, control=path2)

	"""
	fig, ax = plt.subplots()
	colors = plt.cm.jet(np.linspace(0,1,len(kwargs)))

	for key, value in kwargs.items():

		df = pd.read_csv(value, index_col=None, header=0)
		ax.hist(df['alpha'], bins=30, color=colors[list(kwargs.keys()).index(key)], density=True, label=key)
		ax.legend(loc='upper right')
		ax.set(xlabel=r'alpha)', ylabel='Weight (a.u)')

	plt.show()

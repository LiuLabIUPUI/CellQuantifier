import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.ticker as tkr
import scipy.stats as stats
import pandas as pd
import itertools
import matplotlib as mpl

from matplotlib import rc
from sklearn import mixture
from sklearn import mixture
from scipy import linalg

def plot_gmm_selection(df,
					   cat_col,
					   max_comp=5):

	"""
	Fit a GaussianMixtureModel (GMM) to data iteratively and find best model

	Parameters
	----------
	df : DataFrame object
		The DataFrame containing cat_col and hist_col columns
	cat_col : str
		Column to use for categorical sorting
	max_comp : int
		The maximum number of components to test for the GMM

	"""

	lowest_bic = np.infty
	bic = []
	n_components_range = range(1, max_comp)
	cv_types = ['spherical', 'tied', 'diag', 'full']
	this_df = df[cat_col]
	for cv_type in cv_types:
		for n_components in n_components_range:
			# Fit a Gaussian mixture with EM
			f = np.ravel(this_df).astype(np.float)
			f = f.reshape(-1,1)
			gmm = mixture.GaussianMixture(n_components=n_components,
										  covariance_type=cv_type)
			gmm.fit(f)
			bic.append(gmm.bic(f))
			if bic[-1] < lowest_bic:
				lowest_bic = bic[-1]
				best_gmm = gmm

	bic = np.array(bic)
	color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
								  'darkorange'])
	clf = best_gmm
	bars = []

	# Plot the BIC scores
	plt.figure(figsize=(8, 6))
	spl = plt.subplot(1, 1, 1)
	for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
		xpos = np.array(n_components_range) + .2 * (i - 2)
		bars.append(plt.bar(xpos, bic[i * len(n_components_range):
									  (i + 1) * len(n_components_range)],
							width=.2, color=color))
	plt.xticks(n_components_range)
	plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
	plt.title('BIC score per model')
	xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
		.2 * np.floor(bic.argmin() / len(n_components_range))
	plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
	spl.set_xlabel('Number of components')
	spl.legend([b[0] for b in bars], cv_types)

	plt.tight_layout()
	plt.show()


def plot_gmm(df,
	n_comp,
	cat_col,
	hist_col,
	cv_type='full',
	pltshow=True):

	"""
	Fit a GaussianMixtureModel (GMM) to data

	Parameters
	----------
	df : DataFrame object
		The DataFrame containing cat_col and hist_col columns
	n_comp : int
		Number of gaussian components in the GMM
	cat_col : str
		Column to use for categorical sorting
	hist_col : int
		Column containing the data to be fit
	cv_type: str, optional
		The type of covariance to use
	pltshow : bool, optional
		Whether to show the figure or just save to disk


	Examples
	--------

	>>> import pandas as pd
	>>> from cellquantifier.plot import plot_gmm
	>>> path = 'cellquantifier/data/test_physDataMerged.csv'
	>>> df = pd.read_csv(path, index_col=None, header=0)
	>>> plot_gmm(df, 5, 'exp_label', 'D')

	"""

	cats = df[cat_col].unique()
	fig,ax = plt.subplots(len(cats),len(cats))
	colors = plt.cm.jet(np.linspace(0,1,n_comp))

	for i, cat in enumerate(cats):

		this_df = df.loc[df[cat_col] == cat]
		this_df = this_df['D']

		f = np.ravel(this_df).astype(np.float)
		f = f.reshape(-1,1)
		g = mixture.GaussianMixture(n_components=n_comp,covariance_type=cv_type)
		g.fit(f)
		bic = g.bic(f)

		gmm_df = pd.DataFrame()
		gmm_df['weights'] = g.weights_
		gmm_df['means'] = g.means_
		gmm_df['covar'] = g.covariances_.reshape(1,n_comp)[0]
		gmm_df = gmm_df.sort_values(by='means')

		f_axis = f.copy().ravel()
		f_axis.sort()

		ax[0,i].hist(f, bins=100, histtype='bar', density=True, ec='red', alpha=0.5)
		ax[0,i].set_title(cat)
		ax[1,i].pie(gmm_df['weights'], colors=colors, autopct='%1.1f%%')

		for j in range(n_comp):
			ax[0,i].plot(f_axis,gmm_df['weights'].to_numpy()[j]*stats.norm.pdf(f_axis,gmm_df['means'].to_numpy()[j],\
						 np.sqrt(gmm_df['covar'].to_numpy()[j])).ravel(), c=colors[j])

	plt.rcParams['agg.path.chunksize'] = 10000
	plt.grid()
	plt.tight_layout()

	if pltshow:
		plt.show()

	return g

from matplotlib import rc
from sklearn import mixture
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.ticker as tkr
import scipy.stats as stats
import pandas as pd


def plot_gmm(df,
			n_comp,
			cat_col,
			hist_col,
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
		g = mixture.GaussianMixture(n_components=n_comp,covariance_type='full')
		g.fit(f)

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

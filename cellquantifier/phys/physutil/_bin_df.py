import pandas as pd
import numpy as np

def bin_df(df, bin_col, nbins=10):

	"""
	Add 'category' column to dataframe by binning w.r.t bin_col

	Parameters
	----------

	df : DataFrame
		DataFrame

	bin_col : str,
		column in df to bin by

	nbins : int,
		number of bins

	"""

	df['category'] = pd.cut(df[bin_col], nbins)

	r_min = df[bin_col].to_numpy().min()
	r_max = df[bin_col].to_numpy().max()
	bin_size = (r_max-r_min)/nbins
	hist, bin_edges = np.histogram(df[bin_col], nbins) #get bin edges
	bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1]) #get bin centers

	return bin_centers, df

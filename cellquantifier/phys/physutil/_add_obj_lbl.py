import pandas as pd
from skimage.measure import label

def add_obj_lbl(df, mask, is_labeled=True, col_name='region_label'):

	"""
	Add object_lbl column to DataFrame

	Parameters
	----------
	df : DataFrame
		DataFrame with x,y columns

	"""

	if not is_labeled:
		mask = label(mask)

	for i, row in df.iterrows():
		df.at[i, col_name] = int(mask[int(round(df.at[i, 'x'])), \
									    int(round(df.at[i, 'y']))])
	return df

from skimage.util import img_as_ubyte
from skimage.io import imread, imsave
from _add_obj_lbl import add_obj_lbl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def add_obj_type(mask_arr, blobs_df, col_name='region_type', obj_names=None):

	"""

	Adds a obj type column to blobs_df. Each unique obj could be
	cytoplasm, nucleus, etc.

	Pseudo code
	----------
	1. Create region type mask
	2. Call add_obj_lbl() where region labels are encoded region types
	3. Convert encoded region types to preferred names

	Parameters
	----------

	mask_arr : list,
		list of masks each defining a unique obj
	blobs_df: DataFrame
		dataframe containing detection instances and coordinates
	col_name : str, optional
		name to use for region type column
	obj_names: list, optional
		names to use for different regions (in same order as mask_arr)

	Example
	----------
	#read statements
	blobs_df = add_obj_type([parent_mask, child_mask], blobs_df=blobs_df,
							 obj_names=['cyto', 'nuc'])

	"""

	reg_type_mask = np.zeros_like(mask_arr[0])
	for i, mask in enumerate(mask_arr):
		reg_type_mask[mask > 0] = i + 1

	blobs_df = add_obj_lbl(blobs_df, reg_type_mask, col_name=col_name)

	if obj_names:
		num_types = blobs_df['region_type'].unique()
		for i, num_type in enumerate(num_types):
			blobs_df.loc[blobs_df['region_type'] == num_type, 'region_type'] = \
										obj_names[i]

	return blobs_df

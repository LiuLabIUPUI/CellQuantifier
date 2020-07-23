import pandas as pd
import numpy as np
import cv2
from skimage.measure import regionprops

def color_regions(mask, int_df, typ_arr, typ_clr_arr=None):

	"""

	Generate an image where regions are classified and colored
	according to their class

	Pseudo code
	----------
	1. Initialize RGB mask
	2. Get unique labels for each region type
	3. Color code the RGB mask

	Parameters
	----------
	mask : ndarray,
		Mask to be transformed to color
	int_df: DataFrame,
		Intensity DataFrame
	typ_arr: list,
		List of different region types e.g. [type1, type2, ...]
	typ_clr_rr: list,
		List of colors to use for each type.

	Returns
	-------
	mask_rgb: ndarray,
		Color coded mask

	"""

	tmp = np.zeros_like(mask)
	mask_rgb = np.dstack((tmp, tmp, tmp))

	for i, typ in enumerate(typ_arr):
		lbl_by_type = int_df.loc[int_df['cell_type'] == typ, 'label'].unique()
		for lbl in lbl_by_type:
			typ_mask = np.where(mask == lbl)
			mask_rgb[:,:][typ_mask] = typ_clr_arr[i]

	return mask_rgb

def add_region_class(int_df,
					 cls_div_mat=[[1,.1],[1,-.1]],
					 min_class_int=[.1,.1]):

	"""

	Perform classification based on mean channel intensities

	Pseudo code
	----------
	1. Add cell types to DataFrame

	Parameters
	----------

	int_df : DataFrame
		DataFrame containing avg intensity and label columns

	cls_div_mat: ndarray, optional
		function parameters for partitioning intensity space
		(two linear functions, one each row)

	min_class_int: ndarray, optional
		lowest avg intensity that will be classified

	Returns
	-------
	int_df : DataFrame
		DataFrame with added cell_type column

	"""

	cols = [col for col in int_df.columns if 'avg_' in col]

	for col in cols:
		max = int_df[col].max()
		int_df[col] = int_df[col]/max

	m1,b1 = cls_div_mat[0]; m2,b2 = cls_div_mat[1]

	cols = [col for col in int_df.columns if 'avg_' in col]

	int_df.loc[int_df[cols[1]] > m1*int_df[cols[0]] + b1, 'cell_type'] = 'type1'
	int_df.loc[int_df[cols[1]] < m2*int_df[cols[0]] + b2, 'cell_type'] = 'type2'

	int_df.loc[(int_df[cols[1]] > m2*int_df[cols[0]] + b2) &\
			   (int_df[cols[1]] < m1*int_df[cols[0]] + b1), 'cell_type'] = 'type3'

	int_df.loc[(int_df[cols[0]] < min_class_int[0]) &\
			   (int_df[cols[1]] < min_class_int[1]), 'cell_type'] = 'type4'


	return int_df

def get_int_df(im_arr, mask, im_names=['ch1', 'ch2']):

	"""
	Pseudo code
	----------
	1. Extract region properties from each intensity image
	2. Build intensity dataframe from region properties

	Parameters
	----------

	im_arr : list,
		List of intensity images

	mask : ndarray,
		Mask with optionally labeled regions

	im_names : list,
		Identifiers of channels to use in output DataFrame

	"""

	df_arr = []
	for i, im in enumerate(im_arr):
		prop = regionprops(mask, intensity_image=im)
		prop = [[p.label, p.mean_intensity] for p in prop]
		df = pd.DataFrame(prop, columns=['label','avg_%s_intensity' \
													% (im_names[i])])
		df_arr.append(df)


	int_df = pd.concat(df_arr, axis=1)
	int_df = int_df.loc[:,~int_df.columns.duplicated()]

	return int_df

def dof(im, winh=10, winw=10, stride=10):

	"""Calculates the degree of focus (DOF) of an image over
	   local regions by using a sliding window approach and the
	   variance of laplacian operator

	Parameters
	----------
	im : 2d/3d ndarray
	winh : float, optional
		Height of sliding window
	winhw : float, optional
		Width of sliding window
	stride : int, optional
		Stride for sliding window. Set to winw for non-overlapping windows


	Returns
	-------
	dof_im : 2d/3d ndarray
		The degree of focus image

	"""

	def sliding_window(im, stride, win_size):
		for x in range(0, im.shape[0], stride):
			for y in range(0, im.shape[1], stride):
				yield (x, y, im[y:y + win_size[1], x:x + win_size[0]])


	if len(im.shape) == 2:
		im = im.reshape((1,) + im.shape)

	dof_im = np.zeros_like(im)
	win = sliding_window(im, stride=stride, win_size=(winw, winh))
	for i in range(len(im)):
		for (x, y, window) in win:
			dof = np.std(cv2.Laplacian(window, cv2.CV_64F)) ** 2
			dof_im[y:y+winw, x:x+winh] = dof

	return dof_im

def dof_mask(mask, dof_im):

	"""Calculates the average degree of focus (DOF) over masked
	   regions, replacing object labels with the average

	Parameters
	----------
	mask : 2d/3d ndarray

	dof_im : float, optional
		Height of sliding window

	Returns
	-------
	dof_im : 2d/3d ndarray
		The degree of focus mask

	"""

	dof_im[binary == 0] = 0
	props = regionprops(mask, dof_im)
	for prop in props:
		mask[mask == prop.label] = prop.mean_intensity

	return dof_im

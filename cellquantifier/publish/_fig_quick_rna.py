import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from cellquantifier.plot.plotutil import *
from skimage.segmentation import mark_boundaries
from copy import deepcopy

def fig_quick_rna(blobs_df,
				  int_df,
				  im,
				  lbl_mask=None,
				  mask_arr=None,
				  outline_arr=None,
				  typ_clr_arr=None):

	"""

	Validation figure for RNA expression analysis

	Pseudo code
	----------
	1. Build the figure
	2. Populate upper right panel with cell classifications
	3. Populate left panel with overlay image and cell classifications
	4. Merge blobs_df and int_df on label column
	5. Generate blox plots with copy number and peak intensity per cell

	Parameters
	----------
	blobs_df : DataFrame
		DataFrame containing x,y,peak, and label columns
	int_df: DataFrame
		Dataframe containing label column and avg intensity columns
		(1 avg intensity column per channel used for cell classification)
	im : ndarray,
		Raw image data to be overlayed
	lbl_mask : ndarray,
		Raw mask to be used for cell classification color-coding
	mask_arr: list
		List of masks e.g. [dapi_mask, cell_mask, etc.]
	outline_arr: list
		List of colors to use for overlay outlines (1 per mask in mask_arr)
	typ_clr_arr: list
		List of colors to use for different cell types

	Notes
	-------
	-Support for 2 channel cell classification only e.g. insulin and glucagon


	Example
	--------
	//insert read statements here
	fig_quick_rna(blobs_df, int_df, im,
				  lbl_mask=cell_mask,
				  mask_arr=[dapi_mask, cell_mask],
				  outline_arr=[(0,0,1),(1,1,1)],
				  typ_clr_arr=[(255,255,255),(255,0,0), (0,255,0), (255,255,0)])

	Returns
	-------
	"""

	# """
	# ~~~~~~~~~~~Build figure (1)~~~~~~~~~~~~~~
	# """


	fig = plt.figure(figsize=(12,7))
	ax0 = plt.subplot2grid((7,12), (0, 0), rowspan=7, colspan=6)
	ax1 = plt.subplot2grid((7,12), (0, 7), rowspan=4, colspan=5)
	ax2 = plt.subplot2grid((7,12), (5, 7), rowspan=2, colspan=2)
	ax3 = plt.subplot2grid((7,12), (5, 10), rowspan=2, colspan=2)


	# """
	# ~~~~~~~~~~~Upper Right Panel (2)~~~~~~~~~~~~~~
	# """

	cols = [col for col in int_df.columns if 'avg_' in col]
	for col in cols:
		int_df[col] = int_df[col]/int_df[col].max()


	#Classify cells by partitioning mean intensity space
	int_df = partition_int_df(ax1, int_df, b=.05, c=.1)

	format_ax(ax1, ax_is_box=False, xlabel=r'$I^{\beta}$',
			  ylabel=r'$I^{\alpha}$', label_fontsize=20,
			  xscale = [0, 1, 1, .1],
			  yscale = [0, 1, 1, .1])

	# """
	# ~~~~~~~~~~~Left Panel (3)~~~~~~~~~~~~~~
	# """

	typ_im = deepcopy(im)
	if mask_arr:
		overlay = im
		for i, mask in enumerate(mask_arr):
			overlay = mark_boundaries(overlay, mask_arr[i],
									  color=outline_arr[i], mode='thick')

	#Color-code different cell types
	typ_arr = int_df['cell_type'].unique()
	tmp = np.zeros_like(im)
	mask_rgb = np.dstack((tmp, tmp, tmp))

	for i, typ in enumerate(typ_arr):
		lbl_by_type = int_df.loc[int_df['cell_type'] == typ, 'label'].unique()
		for lbl in lbl_by_type:
			typ_mask = np.where(lbl_mask == lbl)
			mask_rgb[:,:][typ_mask] = typ_clr_arr[i]


	ax0.imshow(im); ax0.imshow(mask_rgb, alpha=.6); ax0.imshow(overlay, alpha=.4)
	ax0.set_xticks([]); ax0.set_yticks([])
	ax0.scatter(blobs_df['y'], blobs_df['x'], s=2, color='blue')

	# """
	# ~~~~~~~~~~~Merge Blobs/Intensity DataFrames (4)~~~~~~~~~~~~~~
	# """

	blobs_df = pd.merge(blobs_df, int_df, on="label")

	# """
	# ~~~~~~~~~~~Lower Right Panel 1 (5)~~~~~~~~~~~~~~
	# """

	count_df = blobs_df.groupby(['label', \
								 'cell_type']).size().reset_index(name="count")


	count_df_arr = [count_df.loc[count_df['cell_type'] == typ, 'count'] \
					for typ in typ_arr]

	ax2.boxplot(count_df_arr, showfliers=False)
	format_ax(ax2, ax_is_box=False)
	ax2.set_ylabel(r'$\mathbf{Copy-Number}$')

	# """
	# ~~~~~~~~~~~Lower Right Panel 2 (5)~~~~~~~~~~~~~~
	# """

	peak_df_arr = [blobs_df.loc[blobs_df['cell_type'] == typ, 'peak'] \
					for typ in typ_arr]

	ax3.boxplot(peak_df_arr, showfliers=False)
	format_ax(ax3, ax_is_box=False)
	ax3.set_ylabel(r'$\mathbf{Peak-Intensity}$')
	plt.show()


def partition_int_df(ax, int_df, b=0, c=.1):

	"""

	Pseudo code
	----------
	1. Define partition functions
	2. Fill cell class regions on axis
	3. Add cell types to DataFrame

	Parameters
	----------

	ax: Axis object,
		axis to show partitioning of intensity space
	int_df : DataFrame
		DataFrame containing avg intensity and label columns
	b: float, optional
		Determines the width of region of class uncertainty (y = x +/- b)
	c: float, optional
		Lowest average intensity that will still be classified


	Returns
	-------
	int_df : DataFrame
		DataFrame with added cell_type column

	"""

	x = np.linspace(0,1,100)
	tmp1, tmp2 = np.zeros_like(x), np.ones_like(x)
	y1 = x + b; y2 = x - b
	ax.plot(x, y1, x, y2, color='black', linewidth=5)
	ax.fill_between(x, tmp1, y2, color='green')
	ax.fill_between(x, tmp2, y1, color='white')
	ax.fill_between(x, y1, y2, color='yellow')
	rect = Rectangle((0,0), c, c, color='red')
	ax.add_patch(rect)

	cols = [col for col in int_df.columns if 'avg_' in col]

	int_df.loc[int_df[cols[0]] > \
			   int_df[cols[1]] + b, 'cell_type'] = 'type1'
	int_df.loc[int_df[cols[0]] < \
			   int_df[cols[1]] - b, 'cell_type'] = 'type2'

	int_df.loc[(int_df[cols[0]] > int_df[cols[1]] - b) &  \
			   (int_df[cols[0]] < int_df[cols[1]] + b), 'cell_type'] = 'type3'

	int_df.loc[(int_df[cols[0]] < c) &  \
			   (int_df[cols[1]] < c), 'cell_type'] = 'type4'

	ax.scatter(int_df['avg_ins_intensity'], \
				int_df['avg_gluc_intensity'], color='blue', s=5)

	return int_df

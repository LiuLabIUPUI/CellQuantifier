from cellquantifier.phys import add_region_class
from cellquantifier.io import get_files_from_prefix, get_unique_prefixes
from cellquantifier.publish import fig_quick_rna
from cellquantifier.publish import fig_quick_rna2
from cellquantifier.segm import clr_code_mask, get_int_df

from skimage.segmentation import clear_border
from glob import glob
from skimage.io import imread,imsave
from skimage.util import img_as_ubyte

import matplotlib.pyplot as plt
import pandas as pd

input_dir = '/home/clayton/Desktop/200416_HLA-DMB-analysis/HLA-DMB1/'
output_dir = '/home/clayton/Desktop/200416_HLA-DMB-analysis/HLA-DMB1/'

files = glob(input_dir + '*fittData.csv')

for file in files:

	# """
	# ~~~~~~~~~~~Preprocessing~~~~~~~~~~~~~~
	# """

	tags = ['dapi-mask',
			'dapi-raw',
		    'dapi-dilated-mask',
		    'ins-raw',
		    'gluc-raw',
			'hla-raw',
		    'hla-fittData']

	fn_arr = get_files_from_prefix(file, tags=tags)
	im_arr = [img_as_ubyte(imread(fn)) for fn in fn_arr if 'tif' in fn]
	df_arr = [pd.read_csv(fn) for fn in fn_arr if 'csv' in fn]
	mask = clear_border(im_arr[2])
	int_df = get_int_df(im_arr[3:5], mask, im_names=['ins','gluc'])

	# """
	# ~~~~~~~~~~~Processing~~~~~~~~~~~~~~
	# """

	int_df = add_region_class(int_df)
	blobs_df = df_arr[0].rename({'region_label': 'label'}, \
							     axis='columns')

	# """
	# ~~~~~~~~~~~Display validation figure~~~~~~~~~~~~~~
	# """

	fig_quick_rna(blobs_df, int_df,
				  [im_arr[1], im_arr[5]], mask)

	plt.show()

# # """
# # ~~~~~~~~~~~Post processing~~~~~~~~~~~~~~
# # """
#
# prefixes = get_unique_prefixes(input_dir, tag='hla-fittData')
# merged_blobs_df = pd.DataFrame(columns=[])
# merged_int_df = pd.DataFrame(columns=[])
#
# for prefix in prefixes:
#
# 	# """
# 	# ~~~~~~~~~~~Get fittData and format~~~~~~~~~~~~~~
# 	# """
#
# 	blobs_df = pd.read_csv(input_dir + prefix + '_hla-fittData.csv')
# 	int_df = pd.read_csv(input_dir + prefix + '_ins-gluc-intensity-lbld.csv')
# 	blobs_df = blobs_df.rename(columns={'region_label': 'label'})
# 	blobs_df = blobs_df.loc[blobs_df['frame'] == 0]
#
# 	# """
# 	# ~~~~~~~~~~~Add cell_type col to blobs_df~~~~~~~~~~~~~~
# 	# """
#
# 	labels = int_df['label'].unique()
# 	for label in labels:
# 		cell_type = int_df.loc[int_df['label'] == label, \
# 							   'cell_type'].to_numpy()[0]
# 		blobs_df.loc[blobs_df['label'] == label, 'cell_type'] = cell_type
#
# 	# """
# 	# ~~~~~~~~~~~Append dataframe to merged_df~~~~~~~~~~~~~
# 	# """
#
#
# 	blobs_df = blobs_df.assign(prefix=prefix)
# 	int_df = int_df.assign(prefix=prefix)
#
# 	merged_blobs_df = pd.concat([merged_blobs_df, blobs_df], \
# 								axis=0, ignore_index=True)
# 	merged_int_df = pd.concat([merged_int_df, int_df], \
# 							   axis=0, ignore_index=True)
#
#
# fig_quick_rna2(merged_blobs_df, merged_int_df)
# plt.show()

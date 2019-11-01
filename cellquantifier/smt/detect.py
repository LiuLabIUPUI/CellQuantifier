import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib_scalebar.scalebar import ScaleBar
from skimage.feature import blob_log

def detect_blobs(image,
		 min_sigma = 1,
		 max_sigma = 3,
		 num_sigma = 5,
		 threshold = .5,
		 peak_threshold_rel = .1,
		 r_to_sigraw = 3,
		 scatter_detection=False):

	"""
	Detect blobs in time-lapse images

	Parameters
	----------
	image : 2D/3D ndarray
		raw image data
	Returns
	-------
	blobs_df : DataFrame object

	Examples
	--------
	>>> from cellquantifier import data
	>>> from cellquantifier.smt.detect import detect_blobs
	>>> detect_blobs(data.simulated_cell(), threshold=.1)

	         x      y  r
	0    286.0  361.0        7.5
	1    286.0  292.0        4.5
	2    286.0  246.0        3.0
	3    286.0   46.0        4.5
	4    285.0  349.0        6.0
	..     ...    ...        ...
	620   20.0  349.0        3.0
	621   19.0  336.0        9.0
	622   18.0  346.0        3.0
	623   18.0  343.0        3.0
	624   18.0  331.0        4.5

		"""

	blobs_df = pd.DataFrame(columns = ['frame', 'x', 'y', 'r'])

	for i in range(len(image[:])):

		raw_blobs = blob_log(image[i],
				 min_sigma = min_sigma,
				 max_sigma = max_sigma,
				 num_sigma = num_sigma,
				 threshold = threshold)

		df = pd.DataFrame({'frame': i*np.ones((raw_blobs.shape[0],)), 'x': raw_blobs[:, 0], 'y': raw_blobs[:, 1], 'r': raw_blobs[:, 2]})
		df_np = df.to_numpy()

		if scatter_detection:
			show_detection(image[i],df)

		blob_max = []
		for blob in df_np:
			x,y,r = int(blob[1]), int(blob[2]), int(round(blob[3]))
			patch = image[i][x-r:x+r+1, y-r:y+r+1]
			max = patch.max()
			blob_max.append(max)

		df['blob_max'] = blob_max
		threshold_abs = image[i].max() * peak_threshold_rel
		df = df[(df['blob_max'] >= threshold_abs)]

		blobs_df = blobs_df.append(df, ignore_index=True)


	blobs_df = blobs_df[(blobs_df['x'] - blobs_df['r'] > 0) &
				  (blobs_df['x'] + blobs_df['r'] + 1 < image[i].shape[0]) &
				  (blobs_df['y'] - blobs_df['r'] > 0) &
				  (blobs_df['y'] + blobs_df['r'] + 1 < image[i].shape[1])]

	blobs_df['sigma_raw'] = r_to_sigraw*blobs_df['r']
	blobs_df = blobs_df[['frame','x','y','r', 'sigma_raw']]

	return blobs_df


def show_detection(image, df):

		df_np = df.to_numpy()
		fig, ax = plt.subplots()
		ax.imshow(image, cmap='gray')
		font = {'family': 'arial', 'weight': 'bold','size': 16}
		scalebar = ScaleBar(1/25, 'um', location = 'upper right', font_properties=font, box_color = 'black', color='white')
		scalebar.length_fraction = .3
		scalebar.height_fraction = .025
		ax.add_artist(scalebar)
		ax.scatter(df['y'],df['x'], color='red', s = 1)
		for blob in df_np:
			frame, y, x, r = blob
			c = plt.Circle((x, y), r, color='red', linewidth=1, fill=False)
			ax.add_patch(c)
		plt.show()

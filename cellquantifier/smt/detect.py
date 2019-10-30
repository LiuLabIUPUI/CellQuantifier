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
		 patch_size = 3,
		 scatter_detection=False):

	"""
	Detect blobs in 2D images

	Parameters
	----------
	image : 2D ndarray
		raw image data
	Returns
	-------
	blobs_df : DataFrame object

	Examples
	--------
	>>> from skimage import data
	>>> from cq.smt import detect_blobs
	>>> detect_blobs(data.coins(), threshold=.1)

	         x      y  sigma_raw
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

	raw_blobs = blob_log(image,
				 min_sigma = min_sigma,
				 max_sigma = max_sigma,
				 num_sigma = num_sigma,
				 threshold = threshold)

	raw_blobs[:, 2] = patch_size*raw_blobs[:, 2]

	blobs = pd.DataFrame({'x': raw_blobs[:, 0], 'y': raw_blobs[:, 1], 'sigma_raw': raw_blobs[:, 2]})
	blobs = blobs[(blobs['x'] - blobs['sigma_raw'] > 0) &
				  (blobs['x'] + blobs['sigma_raw'] + 1 < image.shape[0]) &
				  (blobs['y'] - blobs['sigma_raw'] > 0) &
				  (blobs['y'] + blobs['sigma_raw'] + 1 < image.shape[1])]

	blobs_np = blobs.to_numpy()
	blob_max = []

	for blob in blobs_np:
		x,y,r = int(blob[0]), int(blob[1]), int(round(blob[2]))
		patch = image[x-r:x+r+1, y-r:y+r+1]
		max = patch.max()
		blob_max.append(max)


	blobs['blob_max'] = blob_max
	threshold_abs = image.max() * peak_threshold_rel
	blobs = blobs[(blobs['blob_max'] >= threshold_abs)]

	if scatter_detection:

		blobs_np = blobs.to_numpy()
		fig, ax = plt.subplots()
		ax.imshow(image, cmap='gray')
		font = {'family': 'arial', 'weight': 'bold','size': 16}
		scalebar = ScaleBar(1/25, 'um', location = 'upper right', font_properties=font, box_color = 'black', color='white')
		scalebar.length_fraction = .3
		scalebar.height_fraction = .025
		ax.add_artist(scalebar)
		ax.scatter(blobs['y'],blobs['x'], color='red', s = 1)
		for blob in blobs_np:
			y, x, r, i, d = blob
			c = plt.Circle((x, y), r, color='red', linewidth=1, fill=False)
			ax.add_patch(c)
		plt.show()


	blobs = blobs[['x','y', 'sigma_raw']]

	return blobs

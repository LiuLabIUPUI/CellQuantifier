import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib_scalebar.scalebar import ScaleBar
from skimage.feature import blob_log
from matplotlib_scalebar.scalebar import ScaleBar

def detect_blobs(image, config):

	"""
	Detects blobs in time-course images using a predefined static threshold (same threshold for every image)
	"""

	peak_threshold_rel = config.PEAK_THRESH_REL
	raw_blobs = blob_log(image, min_sigma = config.MIN_SIGMA,
							max_sigma = config.MAX_SIGMA,
							num_sigma = config.NUM_SIGMA,
							threshold=config.THRESHOLD)

	#change blob radius to 3 sigma
	raw_blobs[:, 2] = config.PATCH_SIZE*raw_blobs[:, 2]

	d_to_com = [0 for blob in raw_blobs]
	#filter raw_blobs at or outside the bounds of the image
	blobs = pd.DataFrame({'x': raw_blobs[:, 0], 'y': raw_blobs[:, 1], 'r': raw_blobs[:, 2], 'd_to_com': d_to_com})
	blobs = blobs[(blobs['x'] - blobs['r'] > 0) &
				  (blobs['x'] + blobs['r'] + 1 < image.shape[0]) &
				  (blobs['y'] - blobs['r'] > 0) &
				  (blobs['y'] + blobs['r'] + 1 < image.shape[1])]

	#patch max contains the maximum intensity in the blob detected
	blobs_np = blobs.to_numpy()
	blob_max = []

	for blob in blobs_np:
		x,y,r = int(blob[0]), int(blob[1]), int(round(blob[2]))
		patch = image[x-r:x+r+1, y-r:y+r+1]
		max = patch.max()
		blob_max.append(max)


	blobs['blob_max'] = blob_max
	#filter blobs with intensity lower than 10% of the maximum intensity of the image
	threshold_abs = image.max() * peak_threshold_rel
	blobs = blobs[(blobs['blob_max'] >= threshold_abs)]

	#show the detection if desired
	if config.SCATTER_DETECTION:

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


	#dont return the patch max intensity (for now)
	blobs = blobs[['x','y', 'r']]
	return blobs

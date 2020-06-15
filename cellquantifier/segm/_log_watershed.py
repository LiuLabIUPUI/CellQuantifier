from scipy import ndimage as ndi
from skimage.segmentation import *
from skimage.filters import threshold_otsu
from skimage.feature import blob_log, peak_local_max
from skimage.morphology import erosion, binary_erosion, disk
from copy import deepcopy
from skimage.transform import resize, downscale_local_mean
import numpy as np
import timeit
import matplotlib.pyplot as plt

def log_watershed(image,
				  min_sigma=1,
				  max_sigma=5,
				  num_sigma=10,
				  threshold=0.1,
				  erosion_radius=20,
				  pltshow=False):

	"""Classic watershed algorithm that uses the LoG blob detector
	   to find seeds

	Pseudocode
	----------
	1. Apply otsu threshold to the image
	2. Erode the binary mask
	3. Convert eroded binary mask to distance map
	4. Run LoG blob detector on eroded distance map to find seeds
	5. Run watershed algorithm with seeds

	Parameters
	----------
	image : 2d/3d ndarray
	min_sig : float, optional
		As 'min_sigma' argument for blob_log().
	max_sig : float, optional
		As 'max_sigma' argument for blob_log().
	num_sig : int, optional
		As 'num_sigma' argument for blob_log().
	threshold : float, optional
		Relative threshold for blob_log().
	pltshow : bool, optional
		Whether or not to show the segm result per frame

	Returns
	-------
	mask_arr : 2d/3d ndarray
		The object mask(s)

	"""

	if len(image.shape) == 2:
		image = image.reshape((1,) + image.shape)

	mask_arr = np.zeros_like(image)
	for i in range(image.shape[0]):

		# """
		# ~~~~~~~~~~Get binary mask~~~~~~~~~~~~~~
		# """

		thresh = threshold_otsu(image[i])
		binary = image[i] > thresh

		binary_eroded = binary_erosion(binary, selem=disk(20))
		distance = ndi.distance_transform_edt(binary)
		distance_eroded = ndi.distance_transform_edt(binary_eroded)

		blobs_log = blob_log(distance_eroded,
							 min_sigma=min_sigma,
							 max_sigma=max_sigma,
							 num_sigma=num_sigma,
							 threshold=threshold)

		blobs_log = blobs_log[:, :2]
		tmp = np.zeros_like(binary)
		for peak in blobs_log:
			tmp[int(peak[0]), int(peak[1])] = 255

		markers = ndi.label(tmp)[0]
		mask_arr[i] = watershed(-distance_eroded, markers, mask=binary)

		if pltshow:
			fig, ax = plt.subplots(ncols=3, figsize=(9, 3),
								   sharex=True, sharey=True)

			marked = mark_boundaries(image[i], mask_arr[i])
			ax[0].imshow(marked, cmap=plt.cm.gray)
			ax[1].imshow(-distance, cmap=plt.cm.nipy_spectral)
			ax[1].scatter(blobs_log[:, 1], blobs_log[:, 0])
			ax[2].imshow(mask_arr[i], cmap='coolwarm')
			for a in ax:
				a.set_axis_off()
			plt.show()

	return mask_arr

def log_watershed2(image,
				  min_sigma=1,
				  max_sigma=5,
				  num_sigma=10,
				  threshold=0.1,
				  erosion_radius=20,
				  log_mod=5,
				  pltshow=False):

	"""Classic watershed algorithm that uses the LoG blob detector
	   to find seeds

	Pseudocode
	----------
	1. Apply otsu threshold to the image
	2. Erode the binary mask
	3. Convert eroded binary mask to distance map
	4. Run LoG blob detector on eroded distance map to find seeds
	5. Run watershed algorithm with seeds

	Parameters
	----------
	image : 2d/3d ndarray
	min_sig : float, optional
		As 'min_sigma' argument for blob_log().
	max_sig : float, optional
		As 'max_sigma' argument for blob_log().
	num_sig : int, optional
		As 'num_sigma' argument for blob_log().
	threshold : float, optional
		Relative threshold for blob_log().
	pltshow : bool, optional
		Whether or not to show the segm result per frame

	Returns
	-------
	mask_arr : 2d/3d ndarray
		The object mask(s)

	"""

	if len(image.shape) == 2:
		image = image.reshape((1,) + image.shape)

	# mask_arr = np.zeros_like(image)
	for i in range(image.shape[0]):

		im = image[i]
		print('Segmenting Frame %d/%d' % (i, image.shape[0]))

		# """
		# ~~~~~~~~~~Update markers~~~~~~~~~~~~~~
		# """

		if i % log_mod == 0:

			print('Updating Markers....')
			eroded = erosion(im, selem=disk(20))
			blobs_log = blob_log(eroded,
								 min_sigma=min_sigma,
								 max_sigma=max_sigma,
								 num_sigma=num_sigma,
								 threshold=threshold)
			blobs_log = blobs_log[:, :2]
			tmp = np.zeros_like(eroded)
			for peak in blobs_log:
				tmp[int(peak[0]), int(peak[1])] = 255

			markers = ndi.label(tmp)[0]

		# """
		# ~~~~~~~~~~Get binary mask, distance map~~~~~~~~~
		# """

		thresh = threshold_otsu(im)
		binary = im > thresh
		distance = ndi.distance_transform_edt(binary)

		mask = watershed(-distance, markers, mask=binary)

		if pltshow:
			fig, ax = plt.subplots(ncols=2, figsize=(9, 3),
								   sharex=True, sharey=True)

			marked = mark_boundaries(im, mask)
			ax[0].imshow(marked, cmap=plt.cm.gray)
			ax[0].scatter(blobs_log[:, 1], blobs_log[:, 0])
			ax[1].imshow(mask, cmap='coolwarm')
			for a in ax:
				a.set_axis_off()
			plt.show()

	return mask_arr

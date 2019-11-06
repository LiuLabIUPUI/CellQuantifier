import numpy as np
import matplotlib.pyplot as plt

from ..io import imshow

from skimage.segmentation import clear_border, mark_boundaries
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.filters import unsharp_mask, gaussian
from skimage import img_as_ubyte


def get_mask(image, method, sigma, min_size=50, show_mask=False):

	"""Dispatches segmentation request for a single frame to appropriate function

	Parameters
	----------
	image : 2D ndarray

	method : string
		the method to use when generating the object mask

	Returns
	-------
	mask : 2D ndarray
		The object mask
	"""

	if method == 'label':
		mask = label_image(image)
	elif method == 'threshold':
		mask = threshold(image, sigma, min_size)
	elif method == 'unsharp':
		mask = unsharp(image)

	centroid = regionprops(mask)[0].centroid

	if show_mask:
		fig,ax = plt.subplots(1,2)
		ax[0].imshow(mask, cmap='gray')
		ax[1].imshow(mark_boundaries(image, mask))
		ax[1].scatter(centroid[1],centroid[0])
		plt.show()

	return mask, centroid

def get_mask_batch(image, method, sigma, min_size=50, show_mask=False):

	"""Dispatches segmentation request for a time-series to appropriate function

	Parameters
	----------
	image : 2D ndarray

	method : string
		The method to use when generating the object mask

	Returns
	-------
	mask : 3D ndarray
		The object mask
	"""
	mask = np.zeros_like(image)
	for i in range(len(image)):
		mask[i], centroid = get_mask(image[i], method, sigma, min_size, show_mask)

	return mask, centroid

def threshold(image, sigma, min_size):

	"""Uses blurring/thresholding to create the object mask

	Parameters
	----------
	image : 2D ndarray

	min_size : int
		The size of the smallest object that should be kept in the mask

	Returns
	-------
	mask : 2D ndarray
		The object mask
	"""

	image = gaussian(image, sigma=sigma)
	t = .1*image.max()
	mask = image > t
	mask = mask.astype(int)
	mask = remove_small_objects(mask, min_size)
	mask = label(mask)

	return mask

def unsharp(image, min_size):

	"""Uses unsharp intensity transformation to create the object mask

	Parameters
	----------
	image : 2D ndarray

	Returns
	-------
	mask : 2D ndarray
		The object mask
	"""

	image = clear_border(image)
	mask = unsharp_mask(image, radius=100, amount=100)
	mask = remove_small_objects(mask, min_size)
	mask = label(mask)

	return mask

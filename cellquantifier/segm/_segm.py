import numpy as np

from skimage.segmentation import clear_border, mark_boundaries
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, remove_small_holes, binary_opening, binary_closing, selem, dilation, erosion, disk
from skimage.filters import unsharp_mask, gaussian
from skimage import img_as_ubyte


def get_mask(image, method, show_mask=False):

	"""Dispatches segmentation request to appropriate function

	Parameters
	----------
	image : 2D ndarray

	method : string
		standard deviation of gaussian kernel used to convolve the image

	Returns
	-------
	mask : 2D ndarray
		The object mask
	"""

	if method == 'label':
		mask = label_image(image)
	elif method == 'threshold':
		mask = threshold(image, min_size)
	elif method == 'watershed':
		mask = watershed(image, min_size, threshold)
	elif method == 'unsharp':
		mask = unsharp(image)

	if show_mask:

		fig,ax = plt.subplots(1,2)
		ax[0].imshow(mask, cmap='gray')
		ax[1].imshow(mark_boundaries(image*3, mask))
		plt.show()

	centroid = regionprops(mask)[0].centroid

	return mask, centroid

def threshold(image, min_size=50):

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

	image = gaussian(image, sigma=3)
	t = .1*image.max()
	mask = image > t
	mask = mask.astype(int)
	mask = label(mask)

	return mask

def unsharp(image):

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

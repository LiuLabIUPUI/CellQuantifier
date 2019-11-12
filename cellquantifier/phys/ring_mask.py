def get_ring_mask(mask):

	from skimage.morphology import binary_erosion
	from cellquantifier.segm.mask import get_thres_mask
	from skimage.morphology import disk
	from copy import deepcopy

	"""Create ring mask via binary erosion

	Parameters
	----------
	image : 2D ndarray

	Returns
	-------
	mask : 2D ndarray
		The object mask
	"""

	ring_mask = deepcopy(mask)

	nrings = 5
	ring_width = 10

	selem = disk(ring_width)

	for i in range(nrings):
	    mask = binary_erosion(mask, selem=selem)
	    ring_mask[mask > 0] = i+2

	return ring_mask

from skimage.filters import gaussian

def get_thres_mask(img, sig, thres_rel=0.2):
    """
    Get a mask based on "gaussian blur" and "threshold".

    Parameters
    ----------
    img : ndarray
        Imaging in the format of 2d ndarray.
    sig : float
        Sigma of gaussian blur
    thres_rel : float, optional
        Relative threshold comparing to the peak of the image.

    Returns
    -------
    mask_array_2d: ndarray
        2d ndarray of 0s and 1s
    """

    img = gaussian(img, sigma=sig)
    img = img > img.max()*thres_rel
    mask_array_2d = img.astype(int)

    return mask_array_2d

def get_ring_mask(mask, nrings=1, ring_width=10):

	from skimage.morphology import binary_erosion
	from cellquantifier.segm.mask import get_thres_mask
	from skimage.morphology import disk
	from copy import deepcopy

	"""Create ring mask via binary erosion

	Parameters
	----------
	mask : 2D binary ndarray

    nrings: int, optional
        The number of rings to generate
    ring_width: int, optional
        The width of each ring

	Returns
	-------
	ring_mask : 2D ndarray
		The ring mask
	"""

	ring_mask = deepcopy(mask)
	selem = disk(ring_width)

	for i in range(nrings):
	    mask = binary_erosion(mask, selem=selem)
	    ring_mask[mask > 0] = i+2

	return ring_mask

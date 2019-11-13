from skimage.filters import gaussian
import numpy as np
from pims import Frame
from skimage.morphology import binary_dilation, binary_erosion, disk


def get_thres_mask(img, sig=3, thres_rel=0.2):
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
    mask_array_2d = img.astype(np.uint8)

    return mask_array_2d


def get_thres_mask_batch(pims_frames, sig=3, thres_rel=0.2):
    """
    Get a mask stack based on "gaussian blur" and "threshold".

    Parameters
    ----------
    pims_frames : pims.Frame object
		3d ndarray in the format of pims.Frame.
    sig : float
        Sigma of gaussian blur
    thres_rel : float, optional
        Relative threshold comparing to the peak of the image.

    Returns
    -------
    masks: pims.Frame object
        3d ndarray of 0s and 1s in the format of pims.Frame.

    Examples
    --------
    import pims
    from cellquantifier.io import imshow
    from cellquantifier.segm import get_thres_mask_batch
    frames = pims.open('cellquantifier/data/simulated_cell.tif')
    masks = get_thres_mask_batch(frames, sig=3, thres_rel=0.1)
    imshow(masks[0], masks[10], masks[20], masks[30], masks[40])
    """

    shape = (len(pims_frames), pims_frames[0].shape[0], pims_frames[0].shape[1])
    masks = Frame(np.zeros(shape, dtype=np.uint8))
    for i in range(len(pims_frames)):
        masks[i] = get_thres_mask(pims_frames[i], sig=sig, thres_rel=thres_rel)

    return masks


def get_dist2boundary_mask(mask):
    """
    Create dist2boundary mask via binary dilation and erosion

	Parameters
	----------
	mask : 2D binary ndarray

	Returns
	-------
	dist_mask : 2D int ndarray
		The distance t0 boundary mask
        if on the mask boundary, the value equals 0.
        if outside of the mask boundary, the value is positive.
        if inside of the mask boundary, the value is negative.

    Examples
	--------
	from cellquantifier.segm.mask import get_thres_mask, get_dist2boundary_mask
    from cellquantifier.io.imshow import imshow
    from cellquantifier.data import simulated_cell

    m = 285; n = 260; delta = 30
    img = simulated_cell()[0][m:m+delta, n:n+delta]
    mask = get_thres_mask(img, sig=1, thres_rel=0.5)
    dist2boundary_mask = get_dist2boundary_mask(mask)
    imshow(dist2boundary_mask)
	"""

    dist_mask = np.zeros(mask.shape, dtype=int)
    selem = disk(1)
    dist_mask[ mask==0 ] = 999999
    dist_mask[ mask==1 ] = -999999

    mask_outwards = mask.copy()
    i = 1
    while True:
        dilated_mask_outwards = binary_dilation(mask_outwards, selem=selem)
        dist_mask[ dilated_mask_outwards ^ mask_outwards==1 ] = i
        mask_outwards = dilated_mask_outwards
        i = i + 1
        if np.count_nonzero(dist_mask == 999999) == 0:
            break

    mask_inwards = mask.copy()
    i = 0
    while True:
        shrinked_mask_inwards = binary_erosion(mask_inwards, selem=selem)
        dist_mask[ shrinked_mask_inwards ^ mask_inwards==1 ] = i
        mask_inwards = shrinked_mask_inwards
        i = i - 1
        if np.count_nonzero(dist_mask == -999999) == 0:
            break

    return dist_mask

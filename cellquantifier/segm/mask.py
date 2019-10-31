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

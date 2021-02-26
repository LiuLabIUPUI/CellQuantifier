from skimage.filters import gaussian
import numpy as np
from skimage.morphology import binary_dilation, binary_erosion, disk
from skimage.filters import threshold_li
from skimage.segmentation import clear_border

def mask_to_3d_coord(mask):

    """
    Get 3d coordinates from mask to plot 3d plot in matplotlib

    Parameters
    ----------
    mask : 2darray
        Imaging in the format of 2d ndarray.

    Returns
    -------
    X, Y, Z: 2darray
        2darray
    """
    X = np.arange(0, mask.shape[1], 1)
    Y = np.arange(0, mask.shape[0], 1)
    Y, X = np.meshgrid(X, Y)
    Z = mask


    Z = Z == 255
    selem = disk(1)
    Z_be = binary_erosion(Z, selem=selem)
    X = X[Z ^ Z_be]
    Y = Y[Z ^ Z_be]
    Z = Z[Z ^ Z_be]


    return X, Y, Z

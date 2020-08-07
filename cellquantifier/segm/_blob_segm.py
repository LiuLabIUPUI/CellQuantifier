import numpy as np
from skimage.io import imsave
from skimage.util import img_as_ubyte
from ..segm import *

def blob_segm(tif, blobs_df, output_path_prefix,
    mask_sig=None,
    mask_thres_rel=None,
    ):
    """
    Segm raw tif file using blob information.

    Pseudo code
    ----------
    1. Iterate the blobs_df by 'particle'
    2. Generate a 3d ndarray with size of 6*r by 6*r
    3. Crop patches from tif and paste in 3d ndarray, starting position 2*r
    4. Save the 3d ndarray as tif in the output_path

    Parameters
    ----------
    tif : 3darray
		3d ndarray.

    blobs_df : DataFrame
		DataFrame contains with columns 'x', 'y', 'r', 'frame', 'particle'

    output_path_prefix : str
		Output path prefix for all sub tif files

    Returns
    -------
    No return values. A bunch of sub tif files segmented from raw tif file.

    Examples
	--------
    """

    # """
    # ~~~~XXX~~~~
    # """
    particles = blobs_df['particle'].unique()

    ind = 1
    tot = len(particles)
    for particle in particles:
        print("Segmenting (%d/%d)" % (ind, tot))
        ind = ind + 1

        curr_df = blobs_df[ blobs_df['particle']==particle ]
        r = int(round(curr_df['r'].max()))
        sub_tif = np.zeros((len(tif), 6*r, 6*r))

        for index in curr_df.index:
            frame = int(curr_df.loc[index, 'frame'])
            x0 = int(curr_df.loc[index, 'x'])
            y0 = int(curr_df.loc[index, 'y'])
            r0 = int(round(curr_df.at[index, 'r']))
            patch = tif[frame][x0-r0:x0+r0+1, y0-r0:y0+r0+1]
            try:
                # sub_tif[frame][2*r:2*r+2*r0+1, 2*r:2*r+2*r0+1] = patch
                sub_tif[frame][3*r-r0:3*r+r0+1, 3*r-r0:3*r+r0+1] = patch
            except:
                pass

        sub_tif = sub_tif / sub_tif.max()
        sub_tif = img_as_ubyte(sub_tif)

        imsave(output_path_prefix + '-' + str(particle) + '.tif', sub_tif)

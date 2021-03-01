import numpy as np; import pandas as pd
from skimage.morphology import binary_dilation, binary_erosion, disk

def count_boundary_foci_num(
    Boundary_mask,
    Boundary_thickness,
    Ch1_df,
    Ch2_df,
    Ch1_label='Ch1',
    Ch2_label='Ch2',
    ):
    """
    Calculate the foci boundary/internal num for every frame

    Parameters
    ----------
    Boundary_mask : ndarray
        Boundary binary mask

    Boundary_mask : int
        How many pixels should the boundary dilate and erode

    Ch1_df : DataFrame
        Channel 1 DataFrame containing 'x', 'y', 'r', 'frame'

    Ch2_df : DataFrame
        Channel 2 DataFrame containing 'x', 'y', 'r', 'frame'

    Ch1_label : str, optional

    Ch2_label : str, optional

    Returns
    -------
    df: DataFrame
        DataFrame containing 'frame', 'Ch1_bdr_num', 'Ch1_itl_num',
        'Ch2_bdr_num', 'Ch2_itl_num'
    """

    cols = ['frame',
            Ch1_label + '_bdr_num',
            Ch1_label + '_itl_num',
            Ch2_label + '_bdr_num',
            Ch2_label + '_itl_num',
            ]
    df = pd.DataFrame([], columns=cols)
    df['frame'] = np.arange(0, len(Boundary_mask))

    selem = disk(Boundary_thickness)
    dilated_mask = np.zeros(Boundary_mask.shape, dtype=Boundary_mask.dtype)
    Internal_mask = np.zeros(Boundary_mask.shape, dtype=Boundary_mask.dtype)
    bdr_mask = np.zeros(Boundary_mask.shape, dtype=Boundary_mask.dtype)
    for i in range(len(dilated_mask)):
        dilated_mask[i] = binary_dilation(Boundary_mask[i], selem=selem)
        Internal_mask[i] = binary_erosion(Boundary_mask[i], selem=selem)
        bdr_mask[i] = dilated_mask[i] ^ Internal_mask[i]

    for i in range(len(Boundary_mask)):
        curr_Ch1_df = Ch1_df[ Ch1_df['frame']==i ]
        curr_Ch2_df = Ch2_df[ Ch2_df['frame']==i ]

        Ch1_bdr_num = 0
        Ch1_itl_num = 0
        for index in curr_Ch1_df.index:
            r = int(round(curr_Ch1_df.loc[index, 'x']))
            c = int(round(curr_Ch1_df.loc[index, 'y']))
            if bdr_mask[i, r, c]>0:
                Ch1_bdr_num = Ch1_bdr_num + 1
            if Internal_mask[i, r, c]>0:
                Ch1_itl_num = Ch1_itl_num + 1

            # Special handling for the top surface
            if i==len(Boundary_mask)-1:
                Ch1_bdr_num = len(curr_Ch1_df)
                Ch1_itl_num = 0

        Ch2_bdr_num = 0
        Ch2_itl_num = 0
        for index in curr_Ch2_df.index:
            r = int(round(curr_Ch2_df.loc[index, 'x']))
            c = int(round(curr_Ch2_df.loc[index, 'y']))
            if bdr_mask[i, r, c]>0:
                Ch2_bdr_num = Ch2_bdr_num + 1
            if Internal_mask[i, r, c]>0:
                Ch2_itl_num = Ch2_itl_num + 1

            # Special handling for the top surface
            if i==len(Boundary_mask)-1:
                Ch2_bdr_num = len(curr_Ch2_df)
                Ch2_itl_num = 0

        df.loc[i, :] = np.array([i, Ch1_bdr_num, Ch1_itl_num,
                                Ch2_bdr_num, Ch2_itl_num])

    df = df.astype(int)

    return df, bdr_mask

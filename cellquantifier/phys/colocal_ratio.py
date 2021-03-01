import numpy as np; import pandas as pd

def calculate_colocal_ratio(Ch1_df, Ch2_df, Ch1_mask,
    Ch1_name='Ch1 colocal ratio',
    Ch2_name='Ch2 colocal ratio',
    ):
    """
    Calculate the foci colocal ratio for every frame

    Parameters
    ----------
    Ch1_df : DataFrame
        Channel 1 DataFrame containing 'x', 'y', 'r', 'frame'

    Ch2_df : DataFrame
        Channel 2 DataFrame containing 'x', 'y', 'r', 'frame'

    Ch1_mask : ndarray
        Channel 1 binary mask

    Ch1_name : str, optional
        Column name of Ch1 coloal ratio in result df

    Ch2_name : str, optional
        Column name of Ch2 coloal ratio in result df

    Returns
    -------
    df: DataFrame
        DataFrame containing 'frame', 'Ch1_name', 'Ch2_name'
    """

    df = pd.DataFrame([], columns=['frame', Ch1_name, Ch2_name])
    df['frame'] = np.arange(0, len(Ch1_mask))

    for i in range(len(Ch1_mask)):
        curr_Ch1_df = Ch1_df[ Ch1_df['frame']==i ]
        curr_Ch2_df = Ch2_df[ Ch2_df['frame']==i ]
        overlap_num = 0
        for index in curr_Ch2_df.index:
            r = int(round(curr_Ch2_df.loc[index, 'x']))
            c = int(round(curr_Ch2_df.loc[index, 'y']))
            if Ch1_mask[i, r, c]>0:
                overlap_num = overlap_num + 1

        try:
            df.loc[i, Ch1_name] = overlap_num / len(curr_Ch1_df)
        except:
            pass

        try:
            df.loc[i, Ch2_name] = overlap_num / len(curr_Ch2_df)
        except:
            pass

        df[Ch1_name] = df[Ch1_name].astype(float)
        df[Ch2_name] = df[Ch2_name].astype(float)

    return df

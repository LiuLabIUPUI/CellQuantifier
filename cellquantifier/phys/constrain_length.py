import pandas as pd
import numpy as np

def add_constrain_length(df, pixel_size):
    """
    Add column to df: 'traj_length'

    Parameters
    ----------
    df : DataFrame
        DataFrame containing 'x', 'y', 'frame', 'particle'

    Returns
    -------
    df: DataFrame
        DataFrame with added 'traj_length' column
    """

    # """
	# ~~~~Initialize df~~~~
	# """
    df = df.sort_values(['particle', 'frame'])
    for col in ['constrain_length']:
        if col in df:
            df = df.drop(col, axis=1)

    print(len(df))

    # """
	# ~~~~calculate~~~~
	# """
    x_mean = (df.groupby('particle')['x'].transform(pd.Series.mean))
    print(len(x_mean))
    y_mean = (df.groupby('particle')['y'].transform(pd.Series.mean))
    df['dx2'] = (df['x'] - x_mean)**2
    df['dy2'] = (df['y'] - y_mean)**2
    dx2_mean = (df.groupby('particle')['dx2'].transform(pd.Series.mean))
    dy2_mean = (df.groupby('particle')['dy2'].transform(pd.Series.mean))
    constrain_length = (dx2_mean + dy2_mean) ** 0.5 * pixel_size
    print(len(constrain_length))
    df['constrain_length'] = constrain_length

    return df

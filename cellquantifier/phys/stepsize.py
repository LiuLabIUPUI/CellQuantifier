import pandas as pd
import numpy as np

def add_stepsize(df, pixel_size):

    # """
	# ~~~~Initialize df~~~~
	# """
    df = df.sort_values(['particle', 'frame'])
    for col in ['stepsize']:
        if col in df:
            df = df.drop(col, axis=1)

    # """
	# ~~~~calculate~~~~
	# """
    delta_x = (df.groupby('particle')['x'].apply(pd.Series.diff))
    df['dx'] = delta_x
    delta_y = (df.groupby('particle')['y'].apply(pd.Series.diff))
    df['dy'] = delta_y
    stepsize = (delta_x**2 + delta_y**2) ** 0.5 * pixel_size

    # """
	# ~~~~filter out 'v' which is not adjacent~~~~
	# """
    delta_frame = (df.groupby('particle')['frame'].apply(pd.Series.diff))
    df['adjacent_frame'] = delta_frame==1
    df['stepsize'] = stepsize[ df['adjacent_frame'] ]

    return df.drop(['dx', 'dy', 'adjacent_frame'], axis=1)

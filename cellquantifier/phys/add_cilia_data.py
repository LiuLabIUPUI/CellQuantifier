import pandas as pd
import numpy as np

def add_dist_to_half_cilia(df):
    x = df['x']
    y = df['y']
    x_g = df['x_global']
    y_g = df['y_global']

    df['dist_to_half_cilia'] = ((x-x_g)**2 + (y-y_g)**2)**0.5
    return df


def add_half_sign(df, flip_sign=False):
    df['half_sign'] = 0

    x0 = df.iloc[0] ['x_global']
    y0 = df.iloc[0] ['y_global']
    x1 = df.iloc[1] ['x_global']
    y1 = df.iloc[1] ['y_global']

    x = df.iloc[0] ['x']
    y = df.iloc[0] ['y']

    if y > (y1-y0)/(x1-x0)*(x-x0) + y0:
        df['half_sign'].iloc[0]  = 1
    else:
        df['half_sign'].iloc[0]  = -1


    for i in range(1, len(df)):
        x0 = df.iloc[i-1] ['x_global']
        y0 = df.iloc[i-1] ['y_global']
        x1 = df.iloc[i] ['x_global']
        y1 = df.iloc[i] ['y_global']

        x = df.iloc[i] ['x']
        y = df.iloc[i] ['y']

        if y > (y1-y0)/(x1-x0)*(x-x0) + y0:
            df['half_sign'].iloc[i]  = 1
        else:
            df['half_sign'].iloc[i]  = -1

    if flip_sign:
        df['half_sign'] = df['half_sign'] * -1

    return df


def add_more_info(df):

    particles = sorted(df['particle'].unique())

    for particle in particles:
        curr_df = df[ df['particle']==particle ]
        height = curr_df['dist_to_half_cilia'] * curr_df['half_sign']
        height_norm = (height - height.min()) / (height.max() - height.min())

        df.loc[df['particle']==particle, 'height'] = height
        df.loc[df['particle']==particle, 'h_norm'] = height_norm

    df = df.sort_values(['particle', 'frame'])
    df['h_norm_diff'] = (df.groupby('particle')['h_norm'].apply(pd.Series.diff))
    df['v_norm'] = df['h_norm_diff'] * df['frame_rate']
    df['v_norm_abs'] = np.abs(df['v_norm'])

    return df

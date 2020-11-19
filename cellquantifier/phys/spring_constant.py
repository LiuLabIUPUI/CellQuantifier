import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys

def model1(df):
    X = df[['X_t']].values
    Y = df[['Y_t']].values

    tmp_df = df[['X_t', 'Y_t']].dropna()
    if not tmp_df.empty:
        tmp_X = tmp_df[['X_t']].values
        tmp_Y = tmp_df[['Y_t']].values
    else:
        tmp_X = np.array([[0]])
        tmp_Y = np.array([[0]])

    coef = float(LinearRegression().fit(tmp_X, tmp_Y).coef_)
    return pd.Series(np.full((len(X)), coef))

def model2(df):
    X = df[['X_t']].values
    Y = df[['Y_t']].values

    tmp_df = df[['X_t', 'Y_t']].dropna()
    if not tmp_df.empty:
        tmp_X = tmp_df[['X_t']].values
        tmp_Y = tmp_df[['Y_t']].values
    else:
        tmp_X = np.array([[0]])
        tmp_Y = np.array([[0]])

    intercept = float(LinearRegression().fit(tmp_X, tmp_Y).intercept_)
    return pd.Series(np.full((len(X)), intercept))


def add_spring_constant(df, pixel_size,
    diagnostic=False):
    """
    Add column to df: 'spring_constant'
    The unit is Kb*T/(nm2)
    reference: https://doi.org/10.1016/j.tig.2019.06.007

    Parameters
    ----------
    df : DataFrame
        DataFrame containing 'x', 'y', 'frame', 'particle'

    Returns
    -------
    df: DataFrame
        DataFrame with added 'spring_constant' column
    """

    # """
	# ~~~~Initialize df~~~~
	# """
    df = df.sort_values(['particle', 'frame'])
    for col in ['spring_constant', 'spring_intercept']:
        if col in df:
            df = df.drop(col, axis=1)

    print(len(df))

    # """
	# ~~~~calculate~~~~
	# """
    x_mean = (df.groupby('particle')['x'].transform(pd.Series.mean))
    df['X_t'] = (df['x'] - x_mean) * pixel_size
    print(len(df['X_t']))

    delta_x = (df.groupby('particle')['x'].apply(pd.Series.diff))
    delta_frame = (df.groupby('particle')['frame'].apply(pd.Series.diff))
    df['adjacent_frame'] = delta_frame==1
    Y_t = delta_x * pixel_size
    df['Y_t'] = Y_t[ df['adjacent_frame'] ]
    print(len(df['Y_t']))

    coef = df.groupby('particle').apply(model1)
    intercept = df.groupby('particle').apply(model2)
    df['spring_constant'] = (coef).to_numpy()
    df['spring_intercept'] = intercept.to_numpy()
    print(len(df['spring_constant']))
    df = df[ df['traj_length']>=80 ]
    print(df[['particle', 'frame', 'X_t', 'Y_t', 'spring_constant']])


    # """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Diagnostic~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # """
    if diagnostic:
        particles = df['particle'].unique()
        for particle in particles:
            curr_df = df[ df['particle']==particle ]

            if curr_df['traj_length'].mean() >= 80:
                fig, ax = plt.subplots(figsize=(9, 9))
                ax.plot(curr_df['X_t'], curr_df['Y_t'], 'o')

                K = curr_df['spring_constant'].mean()
                C = curr_df['spring_intercept'].mean()
                X_t = np.linspace(curr_df['X_t'].min(), curr_df['X_t'].max(), 50)
                Y_t = K * X_t + C
                ax.plot(X_t, Y_t, '-')

                ax.text(0.95,
                        0.00,
                        K,
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        fontsize = 12,
                        color = (1, 1, 1, 0.8),
                        transform=ax.transAxes,
                        )

                plt.show()

    return df

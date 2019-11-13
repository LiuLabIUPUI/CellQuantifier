def add_label(df, region_mask):

    """
    Label particles in a DataFrame based on region derived from region_mask

    Parameters
    ----------
    region_mask : ndarray
        Imaging in the format of 2d ndarray.
    df : DataFrame
        DataFrame containing x,y columns

    Returns
    -------
    df: DataFrame
        DataFrame with added region_label column

    """

    for i, row in df.iterrows():

        df.at[i, 'region_label'] = region_mask[int(round(df.at[i, 'x'])), int(round(df.at[i, 'y']))]

    return df

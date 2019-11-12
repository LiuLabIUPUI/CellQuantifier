def add_label(region_mask, df):

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

        df.at[i, 'region_label'] = region_mask[int(df.at[i, 'x'])][int(df.at[i, 'y'])]

    return df

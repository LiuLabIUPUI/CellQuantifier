def add_foci_num(df):
    """
    Add column to df: 'foci_num'

    Parameters
    ----------
    df : DataFrame
        DataFrame containing 'frame'

    Returns
    -------
    df: DataFrame
        DataFrame with added 'foci_num' column
    """

    frames = df['frame'].unique()

    for frame in frames:
        foci_num = len(df[ df['frame']==frame ])
        df.loc[df['frame']==frame, 'foci_num'] = foci_num

    return df

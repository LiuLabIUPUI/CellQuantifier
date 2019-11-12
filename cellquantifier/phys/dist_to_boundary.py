def add_dist_to_boundary(mask, df):

    """
    Label particles in a DataFrame based on region derived from region_mask

    Parameters
    ----------
    mask : ndarray
        Binary mask
    df : DataFrame
        DataFrame containing x,y columns

    Returns
    -------
    df: DataFrame
        DataFrame with added region_label column

    Examples
    --------
    >>>import pandas as pd
    >>>import pims
    >>>from cellquantifier.phys.dist_to_boundary import add_dist_to_boundary
    >>>from cellquantifier.segm import get_thres_mask, get_ring_mask

    >>>frames = pims.open('cellquantifier/data/simulated_cell.tif')
    >>>mask = get_thres_mask(frames[0], sig=5)
    >>>df = pd.read_csv('cellquantifier/data/test_fittData.csv')
    >>>df = add_dist_to_boundary(mask, df)

    """

    from cellquantifier.segm import add_label, get_dist2bounday_mask

    region_mask = get_dist2bounday_mask(mask)
    df = add_label(region_mask, df)

    return df

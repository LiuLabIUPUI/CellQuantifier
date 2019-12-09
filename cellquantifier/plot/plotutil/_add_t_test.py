import numpy as np
import matplotlib.pyplot as plt
from ...math import t_test


def add_t_test(ax, blobs_df, cat_col, hist_col):

    # """
    # ~~~~~~~~~~~Check if blobs_df is empty~~~~~~~~~~~~~~
    # """

    if blobs_df.empty:
    	return


    # """
    # ~~~~~~~~~~~Prepare the data, category, color~~~~~~~~~~~~~~
    # """

    cats = sorted(blobs_df[cat_col].unique())
    blobs_dfs = [blobs_df.loc[blobs_df[cat_col] == cat] for cat in cats]

    if len(cats) == 2:

        t_stats = t_test(blobs_dfs[0].drop_duplicates('particle')[hist_col],
                        blobs_dfs[1].drop_duplicates('particle')[hist_col])

        t_test_str = 'p-value of t-test: \n%.5f' % (t_stats[1])

        ax.text(0.95,
                0.3,
                t_test_str,
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize = 10,
                color = (0, 0, 0, 1),
                transform=ax.transAxes)

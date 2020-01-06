import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ...math import t_test


def add_t_test(ax, df, cat_col, data_col,
                drop_duplicates=True,
                text_pos=[0.95, 0.3]):

    # """
    # ~~~~~~~~~~~Prepare the data, category, color~~~~~~~~~~~~~~
    # """

    cats = sorted(df[cat_col].unique())
    dfs = [df.loc[df[cat_col] == cat] for cat in cats]

    if len(cats) == 2:
        if drop_duplicates:
            t_stats = t_test(dfs[0].drop_duplicates('particle')[data_col],
                            dfs[1].drop_duplicates('particle')[data_col])
        else:
            t_stats = t_test(cat_col, data_col)

        if t_stats[1] > 0.001:
            t_test_str = 'P = %.3f' % (t_stats[1])
        else:
            t_test_str = 'P = %.2E' % (t_stats[1])

        ax.text(text_pos[0],
                text_pos[1],
                t_test_str,
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize = 13,
                color = (0, 0, 0, 1),
                transform=ax.transAxes)

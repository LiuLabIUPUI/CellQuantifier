import numpy as np; import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp

from ...math import fit_spotcount
from scipy.stats import sem
from ...math import msd, fit_msd

def add_53bp1_count(ax, df,
                    exp_col='exp_label',
                    cell_col='cell_num',
                    cycle_col='cycle_num',
                    RGBA_alpha=0.5,
                    fitting_linewidth=3,
                    elinewidth=None,
                    markersize=None,
                    capsize=2,
                    set_format=True):
    """
    Add mean spot count curve in matplotlib axis.
    For use with cycled imaging only
    The spot counts are obtained from df.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.

    df : DataFrame
		DataFrame containing 'cell_num', 'particle', 'exp_label', 'cycle_num'
        columns

    cat_col : None or str
		Column to use for categorical sorting

    """

    # """
    # ~~~~~~~~~~~Check if df is empty~~~~~~~~~~~~~~
    # """

    if df.empty:
    	return

    # """
    # ~~~~~~~~~~~Divide the data by exp_label~~~~~~~~~~~~~~
    # """

    exps = df[exp_col].unique()
    spot_counts = [[] for exp in exps]
    exp_dfs = [df.loc[df[exp_col] == exp] for exp in exps]

    for i, exp_df in enumerate(exp_dfs):

        # """
        # ~~~~~~~~~~~Divide the data by cell~~~~~~~~~~~~~~
        # """

        cells = exp_df[cell_col].unique()
        cell_dfs = [exp_df.loc[exp_df[cell_col] == cell] for cell in cells]

        for j, cell_df in enumerate(cell_dfs):

        # """
        # ~~~~~~~~~~~Divide the data by cycle~~~~~~~~~~~~~~
        # """

            spot_counts[i].append([])
            cycles = sorted(cell_df[cycle_col].unique())
            cycle_dfs = [cell_df.loc[cell_df[cycle_col] == cycle]\
                                    for cycle in cycles]

            for k, cycle_df in enumerate(cycle_dfs):
                spot_count = cycle_df['particle'].nunique()
                spot_counts[i][j].append(spot_count)

    # """
    # ~~~~~~~~~~Compute the mean and add to axis~~~~~~~~~~~~~~
    # """

        sc_mean = np.mean(spot_counts[i], axis=0)
        sc_mean_fit = fit_spotcount(np.array(cycles), sc_mean)
        yerr = sem(spot_counts[i], axis=0)
        ax.plot(cycles, sc_mean, color='red')
        ax.plot(cycles, sc_mean_fit, color='red')
        ax.errorbar(cycles, sc_mean, yerr=yerr)

    # """
    # ~~~~~~~~~~~Set the label~~~~~~~~~~~~~~
    # """
    if set_format:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.tick_params(labelsize=13, width=2, length=5)

        ax.set_xlabel(r'$\mathbf{Time (s)}$', fontsize=15)
        ax.set_ylabel(r'$\mathbf{Spot Count}$', fontsize=15)

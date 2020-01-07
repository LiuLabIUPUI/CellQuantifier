import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as  np

from cellquantifier.plot.plotutil import *


def add_strip_plot(ax,
                   df,
                   hist_col,
                   cat_col,
                   xlabels=None,
                   ylabel=None,
                   palette=None):

   	"""Generate strip plot

   	Parameters
   	----------

    Parameters
    ----------
    ax : object
        matplotlib axis

    df : DataFrame
		DataFrame containing hist_col, cat_col

    hist_col : str
		Column of df that contains the data

    cat_col : str
		Column to use for categorical sorting

    palette : array
		seaborn color palette

   	"""

    pal = sns.color_palette(palette)

    ax = sns.stripplot(x=cat_col,
                       y=hist_col,
                       data=df,
                       palette=pal,
                       s=3,
                       alpha=.65)

	# """
	# ~~~~~~~~~~~Add the mean bars~~~~~~~~~~~~~~
	# """

    median_width = 0.3
    x = ax.get_xticklabels()

    for tick, text in zip(ax.get_xticks(), x):

        sample_name = text.get_text()
        mean = df[df[cat_col]==sample_name][hist_col].mean()

        ax.plot([tick-median_width/2, tick+median_width/2], [mean, mean],
                lw=3, color='black')


	# """
	# ~~~~~~~~~~~Formatting~~~~~~~~~~~~~~
	# """

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(labelsize=13, width=2, length=5, labelrotation=45.0, axis='x')

    ax.set_ylabel(r'$\mathbf{' + ylabel + '}$', fontsize=15)
    ax.set_xlabel('', fontsize=15)
    ax.set_xlim(-.5,1.5)

    if xlabels:
        ax.set_xticklabels(xlabels)

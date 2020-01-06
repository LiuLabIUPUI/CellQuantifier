import numpy as np
import matplotlib.pyplot as plt

def add_hist(ax,
             blobs_df,
             data_col,
             cat_col=None,
             nbins = 10,
             density=False,
             color='red',
             RGBA_alpha=0.5):
    """
    Add histogram in matplotlib axis.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.
    """


    # """
    # ~~~~~~~~~~~Check if blobs_df is empty~~~~~~~~~~~~~~
    # """

    if blobs_df.empty:
    	return

    # """
    # ~~~~~~~~~~~Prepare the data, category, color~~~~~~~~~~~~~~
    # """

    if cat_col:

        cats = sorted(blobs_df[cat_col].unique())
        blobs_dfs = [blobs_df.loc[blobs_df[cat_col] == cat] for cat in cats]
        colors = plt.cm.jet(np.linspace(0,1,len(cats)))
        colors[:, 3] = RGBA_alpha

        cats_label = cats.copy()
        if 'sort_flag_' in cat_col:
            for m in range(len(cats_label)):
                cats_label[m] = cat_col[len('sort_flag_'):] + ': ' + str(cats[m])

        for i, blobs_df in enumerate(blobs_dfs):
            ax.hist(blobs_df.drop_duplicates(subset='particle')[data_col].to_numpy(),
    					bins=30, color=colors[i], density=True, ec='gray', linewidth=.5,
                        label=cats_label[i] + (r' $\mathbf{(N_{foci} = %d)}$') % blobs_df['particle'].nunique())

    else:
        ax.hist(blobs_df[data_col].to_numpy(),
					bins=30, color=color, density=True)


    # """
    # ~~~~~~~~~~~Set the label~~~~~~~~~~~~~~
    # """

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['left'].set_linewidth(2)
    # ax.spines['bottom'].set_linewidth(2)
    # ax.tick_params(labelsize=13, width=2, length=5)
    # ax.get_yaxis().set_ticks([])

    ax.legend()

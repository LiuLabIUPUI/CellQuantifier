import numpy as np
import matplotlib.pyplot as plt

def add_D_hist(ax, blobs_df, cat_col,
                RGBA_alpha=0.5):
    """
    Add histogram in matplotlib axis.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.

    blobs_df : DataFrame
		DataFrame containing 'particle', 'D' columns

    cat_col : str
		Column to use for categorical sorting

    Returns
    -------
    Add D histograms in the ax.

    Examples
	--------
    import matplotlib.pyplot as plt
    import pandas as pd
    from cellquantifier.plot.plotutil import add_D_hist
    path = 'cellquantifier/data/physDataMerged.csv'
    df = pd.read_csv(path, index_col=None, header=0)
    fig, ax = plt.subplots()
    add_D_hist(ax, df, 'exp_label',
                RGBA_alpha=0.5)
    plt.show()

    """


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
    colors = plt.cm.jet(np.linspace(0,1,len(cats)))
    colors[:, 3] = RGBA_alpha

    cats_label = cats.copy()
    if 'sort_flag_' in cat_col:
        for m in range(len(cats_label)):
            cats_label[m] = cat_col[len('sort_flag_'):] + ': ' + str(cats[m])

    for i, blobs_df in enumerate(blobs_dfs):
        ax.hist(blobs_df.drop_duplicates(subset='particle')['D'].to_numpy(),
					bins=30, color=colors[i], density=True,
                    label=cats_label[i] + ' (foci_num: %d)' % blobs_df['particle'].nunique())

    # """
    # ~~~~~~~~~~~Set the label~~~~~~~~~~~~~~
    # """
    ax.set_ylabel('Frequency')
    ax.set_xlabel('D')
    ax.legend()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def add_D_hist(ax, df,
                cat_col=None,
                RGBA_alpha=0.5,
                set_format=True):
    """
    Add histogram in matplotlib axis.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.

    df : DataFrame
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
    # ~~~~~~~~~~~Check if df is empty~~~~~~~~~~~~~~
    # """

    if df.empty:
    	return


    # """
    # ~~~~~~~~~~~Prepare the data, category, color~~~~~~~~~~~~~~
    # """
    if cat_col:
        cats = sorted(df[cat_col].unique())
        dfs = [df.loc[df[cat_col] == cat] for cat in cats]
        colors = plt.cm.coolwarm(np.linspace(0,1,len(cats)))
        colors[:, 3] = RGBA_alpha

        cats_label = cats.copy()
        if 'sort_flag_' in cat_col:
            for m in range(len(cats_label)):
                cats_label[m] = cat_col[len('sort_flag_'):] + ': ' + str(cats[m])
    else:
        dfs = [df]
        cats_label = ['D']
        colors = [plt.cm.coolwarm(0.99)]

    for i, df in enumerate(dfs):
        sns.set(style="white", palette="coolwarm", color_codes=True)
        sns.distplot(df.drop_duplicates(subset='particle')['D'].to_numpy(),
                    hist=True, kde=True, color=colors[i], ax=ax,
                    label=cats_label[i] + (r' $\mathbf{(N_{foci} = %d)}$') % df['particle'].nunique())
        # ax.hist(df.drop_duplicates(subset='particle')['D'].to_numpy(),
		# 			bins=30, color=colors[i], density=True, ec='gray', linewidth=.5,
        #             label=cats_label[i] + (r' $\mathbf{(N_{foci} = %d)}$') % df['particle'].nunique())

    # """
    # ~~~~~~~~~~~Set the label~~~~~~~~~~~~~~
    # """
    if set_format:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.tick_params(labelsize=13, width=2, length=5)
        ax.get_yaxis().set_ticks([])

        ax.set_ylabel(r'$\mathbf{PDF}$', fontsize=15)
        ax.set_xlabel(r'$\mathbf{D (nm^{2}/s)}$', fontsize=15)
        ax.legend()

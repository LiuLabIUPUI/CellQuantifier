import matplotlib.pyplot as plt
import seaborn as sns

def add_violin(ax, df, data_col, cat_col):
    """
    Add violin plot in matplotlib axis.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.

    df : DataFrame
		DataFrame containing data_col, cat_col

    data_col : str
		Column contain the data

    cat_col : str
		Column to use for categorical sorting

    Returns
    -------
    Add D histograms in the ax.
    """

    cats = sorted(df[cat_col].unique())
    sns.set(style="white", palette="muted", color_codes=True)

    # """
    # ~~~~~~~~~~~Check~~~~~~~~~~~~~~
    # """

    if df.empty or len(cats)!=2:
        print("##############################################")
        print("ERROR: df is empty, or len(cats) is not 2 !!!")
        print("##############################################")
        return

    # """
    # ~~~~~~~~~~~plot violin~~~~~~~~~~~~~~
    # """

    sns.violinplot(x=[cats[0] + ' vs ' + cats[1]]*len(df),
                y=data_col, hue=cat_col,
                data=df, split=True, inner="quartile", ax=ax)

    plt.gca().legend().set_title('')

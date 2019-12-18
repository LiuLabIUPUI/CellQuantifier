import numpy as np; import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp

from ...math import msd, fit_msd

def add_mean_msd(ax, blobs_df, cat_col,
                pixel_size,
                frame_rate,
                divide_num,
                RGBA_alpha=0.5):
    """
    Add mean MSD curve in matplotlib axis.
    The MSD data are obtained from blobs_df.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.

    blobs_df : DataFrame
		DataFrame containing 'particle', 'frame', 'x', and 'y' columns

    cat_col : str
		Column to use for categorical sorting


    Returns
    -------
    Annotate mean MSD in the ax.

    Examples
	--------
	import matplotlib.pyplot as plt
    import pandas as pd
    from cellquantifier.plot.plotutil import add_mean_msd
    path = 'cellquantifier/data/physDataMerged.csv'
    df = pd.read_csv(path, index_col=None, header=0)
    fig, ax = plt.subplots()
    add_mean_msd(ax, df, 'exp_label',
                pixel_size=0.108,
                frame_rate=3.33,
                divide_num=5,
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

        # Calculate individual msd
        im = tp.imsd(blobs_df, mpp=pixel_size, fps=frame_rate, max_lagtime=np.inf)
        #cut the msd curves and convert units to nm

        n = len(im.index) #for use in stand err calculation
        m = int(round(len(im.index)/divide_num))
        im = im.head(m)
        im = im*1e6

        if len(im) > 1:

            # """
            # ~~~~~~~~~~~Plot the mean MSD data and error bar~~~~~~~~~~~~~~
            # """
            imsd_mean = im.mean(axis=1)
            imsd_std = im.std(axis=1, ddof=0)

            #print(imsd_std)
            x = imsd_mean.index.to_numpy()
            y = imsd_mean.to_numpy()
            n_data_pts = np.sqrt(np.linspace(n-1, n-m, m))
            yerr = np.divide(imsd_std.to_numpy(), n_data_pts)
            ax.errorbar(x, y, yerr=yerr, linestyle='None',
                marker='o', color=colors[i])

            # # """
            # # ~~~~~~~~~~~Plot the fit of the average~~~~~~~~~~~~~~
            # # """
            popt_log = fit_msd(x, y, space='log')
            fit_of_mean_msd = msd(x, popt_log[0], popt_log[1])
            ax.plot(x, fit_of_mean_msd, color=colors[i], label=cats_label[i])

    # """
    # ~~~~~~~~~~~Set the label~~~~~~~~~~~~~~
    # """

    ax.set_xlabel(r'$\tau  (\mathbf{s})$')
    ax.set_ylabel(r'$\langle \Delta r^2 \rangle$ ($nm^2$)')
    ax.legend()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ..plot.plotutil import *
from scipy.stats import norm, expon

def fig_quick_lifetimes(
    df, traj_length_thres=20,
    ):

    # """
	# ~~~~Initialize the page layout~~~~
	# """
    # Layout settings
    col_num = 1
    row_num = 1
    divide_index = []
    hidden_index = []
    # Sub_axs_1 settings
    col_num_s1 = 1
    row_num_s1 = 4
    index_s1 = [
        0
        ]
    # Sub_axs_2 settings
    col_num_s2 = 1
    row_num_s2 = 2
    index_s2 = [
        ]

    # """
	# ~~~~Initialize the colors~~~~
	# """
    print("\n")
    print("Preparing colors")
    palette = sns.color_palette('muted')
    c1 = (0, 0, 0, 1)
    c2 = (0, 0, 1, 1)
    c3 = (0, 0, 0, 0.5)
    c4 = (0, 0, 1, 0.5)
    RGBA_alpha = 0.8


    # Layout implementation
    print("\n")
    print("Preparing layout")
    tot_width = col_num * 4
    tot_height = row_num * 3
    all_figures, page = plt.subplots(1, 1, figsize=(tot_width, tot_height))

    grids = []
    axs = []

    axs_s1_bg = []
    axs_s1 = []
    axs_s1_base = []
    axs_s1_slave = []

    axs_s2_bg = []
    axs_s2 = []
    axs_s2_base = []
    axs_s2_slave = []
    for i in range(col_num*row_num):
        r = i // col_num
        c = i % col_num
        w = 1 / col_num
        h = 1 / row_num
        x0 = c * w
        y0 = 1 - (r+1) * h

        # Generate Grids
        grids.append(page.inset_axes([x0, y0, w, h]))

        # Generate individual axs
        axs.append(grids[i].inset_axes([0.33, 0.33, 0.6, 0.6]))

        # Customize axs_s1
        if i in index_s1:
            axs_s1_bg.append(axs[i])
            for i_s1 in range(col_num_s1*row_num_s1):
                r_s1 = i_s1 // col_num_s1
                c_s1 = i_s1 % col_num_s1
                w_s1 = 1 / col_num_s1
                h_s1 = 1 / row_num_s1
                x0_s1 = c_s1 * w_s1
                y0_s1 = 1 - (r_s1+1) * h_s1
                # Generate axs_s1, axs_s1_base, axs_s1_slave
                temp = axs[i].inset_axes([x0_s1, y0_s1, w_s1, h_s1])
                axs_s1.append(temp)
                if y0_s1 == 0:
                    axs_s1_base.append(temp)
                else:
                    axs_s1_slave.append(temp)

        # Customize axs_s2
        if i in index_s2:
            axs_s2_bg.append(axs[i])
            for i_s2 in range(col_num_s2*row_num_s2):
                r_s2 = i_s2 // col_num_s2
                c_s2 = i_s2 % col_num_s2
                w_s2 = 1 / col_num_s2
                h_s2 = 1 / row_num_s2
                x0_s2 = c_s2 * w_s2
                y0_s2 = 1 - (r_s2+1) * h_s2
                # Generate axs_s2, axs_s2_base, axs_s2_slave
                temp = axs[i].inset_axes([x0_s2, y0_s2, w_s2, h_s2])
                axs_s2.append(temp)
                if y0_s2 == 0:
                    axs_s2_base.append(temp)
                else:
                    axs_s2_slave.append(temp)

    # """
	# ~~~~format figures~~~~
	# """
    print("\n")
    print("Formating figures")
    # Format page
    for ax in [page]:
        ax.set_xticks([]);
        ax.set_yticks([])
        format_spine(ax, spine_linewidth=2)

    # Format grids
    for ax in grids:
        ax.set_xticks([]);
        ax.set_yticks([])
        format_spine(ax, spine_linewidth=2)
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_visible(False)

    for i in divide_index:
        for spine in ['bottom']:
            grids[i].spines[spine].set_visible(True)

    # Format axs
    for ax in axs:
        format_spine(ax, spine_linewidth=0.5)
        format_tick(ax, tk_width=0.5)
        format_tklabel(ax, tklabel_fontsize=10)
        format_label(ax, label_fontsize=10)

    for i in hidden_index:
        axs[i].set_xticks([]);
        axs[i].set_yticks([])
        for spine in ['top', 'bottom', 'left', 'right']:
            axs[i].spines[spine].set_visible(False)

    # Format sub_axs_background
    for ax in axs_s1_bg + axs_s2_bg:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_visible(False)

    # Format sub_axs
    for ax in axs_s1 + axs_s2:
        format_spine(ax, spine_linewidth=0.5)
        format_tick(ax, tk_width=0.5)
        format_tklabel(ax, tklabel_fontsize=10)
        format_label(ax, label_fontsize=10)
        ax.set_yticks([])

    # Format sub_axs_slave
    for ax in axs_s1_slave + axs_s2_slave:
        # labels = [item.get_text() for item in ax.get_xticklabels()]
        # empty_string_labels = ['']*len(labels)
        # ax.set_xticklabels(empty_string_labels)
        #
        # labels = [item.get_text() for item in ax.get_yticklabels()]
        # empty_string_labels = ['']*len(labels)
        # ax.set_yticklabels(empty_string_labels)
        ax.set_xticks([])


    # """
	# ~~~~Prepare df for the whole page~~~~
	# """
    print("\n")
    print("Preparing data")
    if not df.empty:
        if 'traj_length' in df:
            df = df[ df['traj_length']>=traj_length_thres ]

        df['dist2bdr_mean'] = (df.groupby('particle')['dist_to_boundary'].transform(pd.Series.mean))
        df = df[ df['dist2bdr_mean']<=-5 ]

        # df['mass_mean'] = df.groupby('particle')['mass'].transform(pd.Series.mean)
        # df['mass_upperlimit'] = df.groupby('raw_data')['mass'].transform(pd.Series.quantile, q=0.5)
        # df = df[ df['mass_mean']<df['mass_upperlimit'] ]

        df = df[ df['traj_length']<=300 ]

        df_WTuw = df[ df['exp_label']=='WTuw' ]
        df_arf6uw = df[ df['exp_label']=='arf6KDuw' ]
        df_WTwo = df[ df['exp_label']=='WTwo' ]
        df_arf6wo = df[ df['exp_label']=='arf6KDwo' ]

    # """
	# ~~~~Save figData~~~~
	# """
    dfp_WTuw = df_WTuw.drop_duplicates('particle')
    dfp_arf6uw = df_arf6uw.drop_duplicates('particle')
    dfp_WTwo = df_WTwo.drop_duplicates('particle')
    dfp_arf6wo = df_arf6wo.drop_duplicates('particle')

    dfp_WTuw = dfp_WTuw[ ['particle', 'traj_length', 'D', 'alpha'] ]
    dfp_arf6uw = dfp_arf6uw[ ['particle', 'traj_length', 'D', 'alpha'] ]
    dfp_WTwo = dfp_WTwo[ ['particle', 'traj_length', 'D', 'alpha'] ]
    dfp_arf6wo = dfp_arf6wo[ ['particle', 'traj_length', 'D', 'alpha'] ]

    dfp_WTuw = dfp_WTuw.rename(columns={'traj_length':'lifetime(frames)'})
    dfp_arf6uw = dfp_arf6uw.rename(columns={'traj_length':'lifetime(frames)'})
    dfp_WTwo = dfp_WTwo.rename(columns={'traj_length':'lifetime(frames)'})
    dfp_arf6wo = dfp_arf6wo.rename(columns={'traj_length':'lifetime(frames)'})

    dfp_WTuw.round(3).to_csv('/home/linhua/Desktop/WT-unwashed.csv', index=False)
    dfp_arf6uw.round(3).to_csv('/home/linhua/Desktop/arf6KD-unwashed.csv', index=False)
    dfp_WTwo.round(3).to_csv('/home/linhua/Desktop/WT-washout.csv', index=False)
    dfp_arf6wo.round(3).to_csv('/home/linhua/Desktop/arf6-washout.csv', index=False)

    # """
	# ~~~~Plot hist~~~~
	# """
    figs = axs_s1
    datas = [
            df_WTuw, df_arf6uw, df_WTwo, df_arf6wo,
            ]
    bins = [
            None, None, None, None,
            ]
    data_cols = [
            'traj_length', 'traj_length', 'traj_length', 'traj_length',
            ]
    colors = [
            c1, c2, c3, c4,
            ]
    legends = [
            'WTuw', 'arf6KDuw', 'WTwo', 'arf6KDwo',
            ]
    for i, (fig, data, bin, data_col, color, legend) \
    in enumerate(zip(figs, datas, bins, data_cols, colors, legends)):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))

        sns.distplot(data[data_col],
                    bins=bin,
                    kde=False,
                    fit=expon,
                    color=color,
                    ax=fig,
                    hist_kws={"alpha": RGBA_alpha,
                    'linewidth': 0.5, 'edgecolor': (0,0,0)},
                    fit_kws={"alpha": RGBA_alpha,
                    'linewidth': 1.5, 'color': color},
                    )

        fig.text(0.95,
                0.2,
                """
                %s lifetime: %.2f
                """ %(legend, expon.fit(data[data_col])[1]),
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize = 6,
                color = (0, 0, 0, 1),
                transform=figs[i].transAxes,
                weight = 'normal',
                )


    # Format scale
    figs = axs_s1
    xscales = [
            [0, 500], [0, 500], [0, 500], [0, 500],
            ]
    yscales = [
            [None, None], [None, None], [None, None], [None, None],
            ]
    for i, (fig, xscale, yscale, ) \
    in enumerate(zip(figs, xscales, yscales,)):
        format_scale(fig,
                xscale=xscale,
                yscale=yscale,
                )

    axs_s1[3].set_xlabel('lifetime (frame)')


    # """
	# ~~~~Save the figure into pdf file, preview the figure in webbrowser~~~~
	# """
    all_figures.savefig('/home/linhua/Desktop/lifetime_figure.pdf', dpi=300)
    plt.clf(); plt.close()

import matplotlib.pyplot as plt
import pandas as pd

from copy import deepcopy
from cellquantifier.plot.plotutil import *


def plot_fig_2(df,
               pixel_size=.1083,
               frame_rate=3.33,
               divide_num=5):

    """
    Construct Figure 2

    Parameters
    ----------

    df : DataFrame
		DataFrame containing 'particle', 'alpha' columns

    pixel_size : float

    frame_rate : float

    divide_num : float

    Example
	--------
    import pandas as pd
    from cellquantifier.publish import plot_fig_2
    from cellquantifier.phys.physutil import add_avg_dist
    from cellquantifier.plot.plotutil import *

    df = pd.read_csv('cellquantifier/data/physDataMerged.csv')
    df = add_avg_dist(df)

    boundary_sorter = [-20, 0]
    bp1_sorter = [-50, 10]

    df['sort_flag_boundary'] = df['avg_dist_bound'].between(boundary_sorter[0], \
                                                            boundary_sorter[1],
                                                            inclusive=True)

    df['sort_flag_53bp1'] = df['avg_dist_53bp1'].between(bp1_sorter[0], \
                                                         bp1_sorter[1],
                                                         inclusive=True)
    plot_fig_2(df)
    """

    fig = plt.figure(figsize=(14,12))
    shape = (16,12)

    # """
    # ~~~~~~~~~~~Initialize Grid~~~~~~~~~~~~~~
    # """

    ax1 = plt.subplot2grid(shape, (0, 0), rowspan=3, colspan=3, projection='polar') #heat maps
    ax2 = plt.subplot2grid(shape, (0, 3), rowspan=3, colspan=3, projection='polar') #heat maps
    ax3 = plt.subplot2grid(shape, (3, 0), rowspan=3, colspan=3, projection='polar') #heat maps
    ax4 = plt.subplot2grid(shape, (3, 3), rowspan=3, colspan=3, projection='polar') #heat maps
    # ax5 = plt.subplot2grid(shape, (0, 4), rowspan=2, colspan=4) #mask fig
    ax6 = plt.subplot2grid(shape, (6, 0), rowspan=8, colspan=4) #ctrl msd curve
    ax7 = plt.subplot2grid(shape, (6, 4), rowspan=4, colspan=2) #ctrl up sp
    ax8 = plt.subplot2grid(shape, (10, 4), rowspan=4, colspan=2) #ctrl down sp
    ax9 = plt.subplot2grid(shape, (6, 6), rowspan=8, colspan=4) #blm msd curve
    ax10 = plt.subplot2grid(shape, (6, 10), rowspan=4, colspan=2) #blm up sp
    ax11 = plt.subplot2grid(shape, (10, 10), rowspan=4, colspan=2) #blm down sp


    # """
    # ~~~~~~~~~~~Get Data~~~~~~~~~~~~~~
    # """

    df_cpy = deepcopy(df)

    # """
    # ~~~~~~~~~~~Heat Maps~~~~~~~~~~~~~~
    # """

    xlabel = r'$\mathbf{D (nm^{2}/s)}$'
    ylabel = r'$\mathbf{D (nm^{2}/s)}$'
    ax = [ax1,ax2]

    add_heat_map(ax,
                 df,
                 'D',
                 'avg_dist_bound',
                 'exp_label',
                  xlabel=xlabel,
                  ylabel=ylabel,
                  nbins=10,
                  hole_size=20)


    xlabel = r'$\mathbf{\alpha}$'
    ylabel = r'$\mathbf{\alpha}$'
    ax = [ax3,ax4]

    add_heat_map(ax,
                 df,
                 'alpha',
                 'avg_dist_bound',
                 'exp_label',
                  xlabel=xlabel,
                  ylabel=ylabel)

    ax1.set_title(r'$\mathbf{BLM}$')
    ax2.set_title(r'$\mathbf{CTRL}$')

    # """
    # ~~~~~~~~~~~CTRL MSD Curve~~~~~~~~~~~~~~
    # """

    ctrl_df = df.loc[df['exp_label'] == 'Ctr']
    ctrl_df_cpy = df_cpy.loc[df_cpy['exp_label'] == 'Ctr']

    add_mean_msd(ax6,
                 ctrl_df_cpy,
                 'sort_flag_boundary',
                 pixel_size,
                 frame_rate,
                 divide_num)

    ax6.set_title(r'$\mathbf{CTRL}$')

    # """
    # ~~~~~~~~~~~CTRL Strip Plots~~~~~~~~~~~~~~
    # """

    add_strip_plot(ax7,
                   ctrl_df,
                   'D',
                   'sort_flag_boundary',
                   xlabels=['Interior', 'Boundary'],
                   ylabel=r'\mathbf{D (nm^{2}/s)}',
                   palette=['blue', 'red'],
                   x_labelsize=8,
                   drop_duplicates=True)

    add_t_test(ax7,
               ctrl_df,
               cat_col='sort_flag_boundary',
               hist_col='D',
               text_pos=[0.9, 0.9])

    ax = add_strip_plot(ax8,
                        ctrl_df,
                        'alpha',
                        'sort_flag_boundary',
                        xlabels=['Interior', 'Boundary'],
                        ylabel=r'\mathbf{\alpha}',
                        palette=['blue', 'red'],
                        x_labelsize=8,
                        drop_duplicates=True)


    add_t_test(ax8,
               ctrl_df,
               cat_col='sort_flag_boundary',
               hist_col='alpha',
               text_pos=[0.9, 0.9])


    # """
    # ~~~~~~~~~~~BLM MSD Curve~~~~~~~~~~~~~~
    # """

    blm_df = df.loc[df['exp_label'] == 'BLM']
    blm_df_cpy = df_cpy.loc[df_cpy['exp_label'] == 'BLM']

    add_mean_msd(ax9,
                 blm_df_cpy,
                 'sort_flag_boundary',
                 pixel_size,
                 frame_rate,
                 divide_num)

    ax9.set_title(r'$\mathbf{BLM}$')

    # """
    # ~~~~~~~~~~~BLM Strip Plot~~~~~~~~~~~~~~
    # """

    add_strip_plot(ax10,
                   blm_df,
                   'D',
                   'sort_flag_boundary',
                   xlabels=['Interior', 'Boundary'],
                   ylabel=r'\mathbf{D (nm^{2}/s)}',
                   palette=['blue', 'red'],
                   x_labelsize=8,
                   drop_duplicates=True)

    add_t_test(ax10,
               blm_df,
               cat_col='sort_flag_boundary',
               hist_col='D',
               text_pos=[0.9, 0.9])

    add_strip_plot(ax11,
                   blm_df,
                   'alpha',
                   'sort_flag_boundary',
                   xlabels=['Interior', 'Boundary'],
                   ylabel=r'\mathbf{\alpha}',
                   palette=['blue', 'red'],
                   x_labelsize=8,
                   drop_duplicates=True)

    add_t_test(ax11,
               blm_df,
               cat_col='sort_flag_boundary',
               hist_col='alpha',
               text_pos=[0.9, 0.9])

    plt.subplots_adjust(wspace=100, hspace=100)
    plt.show()

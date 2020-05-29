import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ..plot.plotutil import *
from ..plot.plotutil._add_mean_msd2 import add_mean_msd2

def fig_quick_antigen_4(
    df=pd.DataFrame([]),
    ):

    # """
	# ~~~~Prepare df for the whole page~~~~
	# """
    print("Preparing data")
    print("\n")
    if not df.empty:
        # """
        # ~~~~traj_length filter~~~~
        # """
        if 'traj_length' in df:
            df = df[ df['traj_length']>=20 ]

        # """
        # ~~~~travel_dist filter~~~~
        # """
        if 'travel_dist' in df:
            travel_dist_min = 0
            travel_dist_max = 7
            df = df[ (df['travel_dist']>=travel_dist_min) & \
            					(df['travel_dist']<=travel_dist_max) ]

        # """
        # ~~~~add particle type filter~~~~
        # """
        if 'particle_type' in df:
        	df = df[ df['particle_type']!='--none--']




        df['date'] = df['raw_data'].astype(str).str[0:6]
        df_mal = df[ df['date'].isin(['200205']) ]
        df_mal_A = df_mal[ df_mal['particle_type']=='A' ]
        df_mal_B = df_mal[ df_mal['particle_type']=='B' ]
        df_cas9 = df[ df['date'].isin(['200220']) ]
        df_cas9_A = df_cas9[ df_cas9['particle_type']=='A' ]
        df_cas9_B = df_cas9[ df_cas9['particle_type']=='B' ]

        # get df_particle, which drop duplicates of 'particle'
        df_particle = df.drop_duplicates('particle')

        dmp = df_mal.drop_duplicates('particle')
        dmp_KN = dmp[ dmp['exp_label']=='MalKN' ]
        dmp_WT = dmp[ dmp['exp_label']=='WT' ]
        dmp_OE = dmp[ dmp['exp_label']=='MalOE' ]

        dmpA = df_mal_A.drop_duplicates('particle')
        dmpA_KN = dmpA[ dmpA['exp_label']=='MalKN' ]
        dmpA_WT = dmpA[ dmpA['exp_label']=='WT' ]
        dmpA_OE = dmpA[ dmpA['exp_label']=='MalOE' ]

        dmpB = df_mal_B.drop_duplicates('particle')
        dmpB_KN = dmpB[ dmpB['exp_label']=='MalKN' ]
        dmpB_WT = dmpB[ dmpB['exp_label']=='WT' ]
        dmpB_OE = dmpB[ dmpB['exp_label']=='MalOE' ]

        df_cas9_particle = df_cas9.drop_duplicates('particle')
        df_cas9_A_particle = df_cas9_A.drop_duplicates('particle')
        df_cas9_B_particle = df_cas9_B.drop_duplicates('particle')


    # """
	# ~~~~Initialize the colors~~~~
	# """
    print("Preparing colors")
    print("\n")
    palette = sns.color_palette('muted')
    c1 = palette[0]
    c2 = palette[1]
    c3 = palette[2]
    # c1 = (0, 0, 1)
    # c2 = (0, 0, 0)
    # c3 = (1, 0, 0)
    RGBA_alpha = 0.4
    c1_alp = (c1[0], c1[1], c1[2], RGBA_alpha)
    c2_alp = (c2[0], c2[1], c2[2], RGBA_alpha)
    c3_alp = (c3[0], c3[1], c3[2], RGBA_alpha)


    # """
	# ~~~~Initialize the page layout~~~~
	# """
    print("Preparing layout")
    print("\n")
    col_num = 3
    row_num = 8
    tot_width = col_num * 4
    tot_height = row_num * 3
    fig, page = plt.subplots(1, 1, figsize=(tot_width, tot_height))

    grids = []
    axs = []
    axs_hist = []
    s_hist = []
    subaxs_base = []
    subaxs_slave = []
    for i in range(col_num*row_num):
        r = i // col_num
        c = i % col_num
        grid_w = 1 / col_num
        grid_h = 1 / row_num
        x_start = c * grid_w
        y_start = 1 - (r+1) * grid_h
        grids.append(page.inset_axes([x_start, y_start, grid_w, grid_h]))
        axs.append(grids[i].inset_axes([0.35, 0.35, 0.6, 0.6]))
        if i%col_num!=0 and i not in [9,10,11,21,22,23]:
            axs_hist.append(axs[i])
            ratio = 1
            temp = axs[i].inset_axes([0, 2*grid_w, 1, grid_w*ratio])
            subaxs_slave.append(temp)
            s_hist.append(temp)
            temp = axs[i].inset_axes([0, grid_w, 1, grid_w*ratio])
            subaxs_slave.append(temp)
            s_hist.append(temp)
            temp = axs[i].inset_axes([0, 0, 1, grid_w*ratio])
            subaxs_base.append(temp)
            s_hist.append(temp)



    for axis in grids:
        for spine in ['left', 'right']:
            axis.spines[spine].set_visible(False)
    for axis in axs_hist:
        for spine in ['top', 'bottom', 'left', 'right']:
            axis.spines[spine].set_visible(False)

    for axis in grids + [page] + axs_hist:
        axis.set_xticks([])
        axis.set_yticks([])

    for axis in subaxs_slave:
        labels = [item.get_text() for item in axis.get_xticklabels()]
        empty_string_labels = ['']*len(labels)
        axis.set_xticklabels(empty_string_labels)


    # """
	# ~~~~Plot msd~~~~
	# """
    figs = [axs[0], axs[3], axs[6], axs[12], axs[15], axs[18]]
    datas = [df_mal, df_mal_A, df_mal_B, df_cas9, df_cas9_A, df_cas9_B]
    order1 = ['MalKN', 'MalOE', 'WT']
    order2 = ['Cas9C1', 'Cas9L1', 'Cas9P3']
    cat_orders = [order1, order2]

    for i in range(len(figs)):
        print("\n")
        print("Plotting (%d/%d)" % (i+1, len(figs)))

        add_mean_msd2(figs[i], datas[i],
                    pixel_size=0.163,
                    frame_rate=2,
                    divide_num=5,

                    cat_col='exp_label',
                    cat_order=cat_orders[i//3],
                    color_order=[c1, c2, c3],
                    RGBA_alpha=1,

                    fitting_linewidth=1.5,
                    )
        set_xylabel(figs[i],
                    xlabel='Time (s)',
                    ylabel=r'MSD (nm$^2$)',
                    )

    # # """
	# # ~~~~Plot hist~~~~
	# # """
    # figs = [s_hist[0], s_hist[1], s_hist[2],
    #         s_hist[3], s_hist[4], s_hist[5]]
    # datas = [dmp_KN, dmp_WT, dmp_OE,
    #         dmp_KN, dmp_WT, dmp_OE,
    #         ]
    # data_cols = ['D', 'D', 'D',
    #         'alpha', 'alpha', 'alpha',
    #         ]
    # colors = [c1, c2, c3,]
    #
    # for i in range(len(figs)):
    #     print("\n")
    #     print("Plotting (%d/%d)" % (i+1, len(figs)))
    #
    #     sns.distplot(datas[i][data_cols[i]],
    #                 kde=False,
    #                 color=colors[i%3],
    #                 ax=figs[i],
    #                 )
    #     # set_xylabel(figs[i],
    #     #             xlabel='Time (s)',
    #     #             ylabel=r'MSD (nm$^2$)',
    #     #             )




    # """
	# ~~~~format figures~~~~
	# """
    print("Formating figures")
    print("\n")
    for ax in axs + s_hist + subaxs_base + subaxs_slave:
        format_spine(ax, spine_linewidth=0.5)
        format_tick(ax, tk_width=0.5)
        format_tklabel(ax, tklabel_fontsize=10)
        format_label(ax, label_fontsize=12)

    for ax in [axs[0], axs[3], axs[6], axs[12], axs[15], axs[18]]:
        format_legend(ax,
                show_legend=True,
                legend_loc='lower right',
                legend_fontweight=6,
                legend_fontsize=6,
                )

    # figs = [fig1]
    # xscales = [[0, 30, 5], ]
    # yscales = [[125, 350, 50], ]
    # for i in range(len(figs)):
    #     format_scale(figs[i],
    #             xscale=xscales[i],
    #             yscale=yscales[i],
    #             )







    # """
	# ~~~~Save the figure into pdf file, preview the figure in webbrowser~~~~
	# """
    fig.savefig('/home/linhua/Desktop/Figure_1.pdf', dpi=300)
    # import webbrowser
    # webbrowser.open_new(r'/home/linhua/Desktop/Figure_1.pdf')
    plt.clf(); plt.close()

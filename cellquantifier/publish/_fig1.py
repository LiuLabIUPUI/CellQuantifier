import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from ..plot.plotutil import *



def plot_fig_1(df):

    # """
	# ~~~~~~~~~~~Prepare the data~~~~~~~~~~~~~~
	# """
    df = pd.read_csv('/home/linhua/Desktop/dutp_paper/fig1/BLM2.csv')
    img = imread('/home/linhua/Desktop/dutp_paper/fig1/BLM2.tif')[0]
    df = df[ df['traj_length']>50 ]
    img_53bp1 = imread('/home/linhua/Desktop/dutp_paper/fig1/53bp1channel.jpg')
    img_dutp = imread('/home/linhua/Desktop/dutp_paper/fig1/dutpchannel.jpg')
    img_comb = imread('/home/linhua/Desktop/dutp_paper/fig1/composite.jpg')


    # """
	# ~~~~~~~~~~~Initialize Grid~~~~~~~~~~~~~~
	# """
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))#, dpi=300)
    ax.set_xticks([]); ax.set_yticks([])
    ax3_1 = ax.inset_axes([0, 0, 0.2, 0.2])
    ax3_2 = ax.inset_axes([0.2, 0, 0.2, 0.2])
    ax3_3 = ax.inset_axes([0.4, 0, 0.2, 0.2])
    ax2_3 = ax.inset_axes([0, 0.2, 0.1, 0.1])
    ax2_2 = ax.inset_axes([0, 0.3, 0.1, 0.1])
    ax2_1 = ax.inset_axes([0, 0.4, 0.1, 0.1])
    ax2_4 = ax.inset_axes([0.2, 0.2, 0.4, 0.4])
    ax2_4a = ax2_4.inset_axes([0.05, 0.65, 0.3, 0.3])
    ax2_4b = ax2_4.inset_axes([0.65, 0.05, 0.3, 0.3])


    # """
	# ~~~~~~~~~~~Add plots to the grid~~~~~~~~~~~~~~
	# """
    ax2_1.imshow(img_53bp1, aspect='equal')
    ax2_1.set_xticks([]); ax2_1.set_yticks([])

    ax2_2.imshow(img_dutp, aspect='equal')
    ax2_2.set_xticks([]); ax2_2.set_yticks([])

    ax2_3.imshow(img_comb, aspect='equal')
    ax2_3.set_xticks([]); ax2_3.set_yticks([])

    anno_traj(ax2_4, df, img,
                show_traj_num=False,
                pixel_size=0.108,
                cb_min=0,
                cb_max=2500,
                cb_major_ticker=500,
                cb_minor_ticker=500,
                show_particle_label=False,
                choose_particle=None,
                show_colorbar=True)


    anno_traj(ax2_4a, df, img,
                show_traj_num=False,
                pixel_size=0.108,
                scalebar_fontsize='small',
                cb_min=0,
                cb_max=2500,
                cb_major_ticker=500,
                cb_minor_ticker=500,
                show_particle_label=False,
                choose_particle=85,
                show_colorbar=False)


    anno_traj(ax2_4b, df, img,
                show_traj_num=False,
                pixel_size=0.108,
                scalebar_fontsize='small',
                cb_min=0,
                cb_max=2500,
                cb_major_ticker=500,
                cb_minor_ticker=500,
                show_particle_label=False,
                choose_particle=55,
                show_colorbar=False)


    add_mean_msd(ax3_1, df,
                cat_col=None,
                pixel_size=0.108,
                frame_rate=3.33,
                divide_num=5,
                RGBA_alpha=0.5,
                set_format=False)

    add_D_hist(ax3_2, df,
                cat_col=None,
                RGBA_alpha=0.5,
                set_format=False)

    add_D_hist(ax3_3, df,
                cat_col=None,
                RGBA_alpha=0.5,
                set_format=False)


    # """
	# ~~~~Save the figure into pdf file, preview the figure in webbrowser~~~~~~~
	# """
    fig.savefig('/home/linhua/Desktop/Figure_1.pdf')
    import webbrowser
    webbrowser.open_new(r'/home/linhua/Desktop/Figure_1.pdf')
    plt.clf(); plt.close()

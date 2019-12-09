from .plotutil import *

def plot_phys_1(blobs_df,
                cat_col,

                pixel_size,
                frame_rate,
                divide_num,

                RGBA_alpha=0.5):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    add_mean_msd(ax[0],
                blobs_df=blobs_df,
                cat_col=cat_col,
                pixel_size=pixel_size,
                frame_rate=frame_rate,
                divide_num=divide_num,
                RGBA_alpha=RGBA_alpha
                )
    ax[0].legend(loc='upper right')



    add_D_hist(ax[1],
                blobs_df=blobs_df,
                cat_col=cat_col,
                RGBA_alpha=RGBA_alpha)
    add_t_test(ax[1],
                blobs_df=blobs_df,
                cat_col=cat_col,
                hist_col=['D'])
    add_gmm(ax[1],
                blobs_df=blobs_df,
                cat_col=cat_col,
                n_comp=3,
                hist_col='D',
                RGBA_alpha=RGBA_alpha)
    ax[1].legend(loc='upper right')




    add_alpha_hist(ax[2],
                blobs_df=blobs_df,
                cat_col=cat_col,
                RGBA_alpha=RGBA_alpha)
    add_t_test(ax[2],
                blobs_df=blobs_df,
                cat_col=cat_col,
                hist_col=['alpha'])
    add_gmm(ax[2],
                blobs_df=blobs_df,
                cat_col=cat_col,
                n_comp=1,
                hist_col='alpha',
                RGBA_alpha=RGBA_alpha)
    ax[2].legend(loc='upper right')


    plt.tight_layout()
    plt.show()

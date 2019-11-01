import numpy as np
import pandas as pd
from cellquantifier.math.gaussian_2d import (gaussian_2d, get_moments,
                                            fit_gaussian_2d)
from cellquantifier.util.plot_util.anno import anno_scatter, anno_blob
from cellquantifier.util.plot_util._plot_end import plot_end


def psf_fit(pims_frame,
            blobs_df,
            diagnostic=False,
            pltshow=False,
            diag_max_dist_err=1,
            diag_max_sig_to_sigraw = 2,
            truth_df=None,
            segm_df=None):
    """
    Point spread function fitting for each frame.

    Parameters
    ----------
    pims_frame : pims.Frame object
        Each frame in the format of pims.Frame.
    bolbs_df : DataFrame
        bolb_df with columns of 'x', 'y', 'r'.
    diagnostic : bool, optional
        If true, print the diagnostic strings.
    pltshow : bool, optional
        If true, show diagnostic plot.
    diag_max_dist_err : float, optional
        Virtual max_dist_err filter.
    diag_max_sig_to_sigraw : float, optional
        Virtual diag_max_sig_to_sigraw filter.
    truth_df : DataFrame, optional
        Ground truth DataFrame with columns of 'x', 'y'.
    segm_df : DataFrame, optional
        Segmentation DataFrame with columns of 'x', 'y'.

    Returns
    -------
    psf_df : DataFrame
        columns=['frame', 'x_raw', 'y_raw', 'r',
                'A', 'x', 'y', 'sig_x', 'sig_y', 'phi','area', 'mass',
                'dist_err', 'sigx_to_sigraw', 'sigy_to_sigraw']
    plt_array_2d :  2d ndarray
        2D ndarray of diagnostic plot.

    Examples
    --------
    import os as _os
    import os.path as osp
    data_dir = osp.abspath(osp.dirname(__file__))
    fname = _os.path.join(data_dir,
               'nubo/data/nucleosome.tif')

    import pims
    from nubo.segm import blob_locate
    frames = pims.open(fname)
    f, plt_segm = blob_locate(frames[0],
                    min_sigma=1,
                    max_sigma=2,
                    num_sigma=5,
                    blob_threshold=0.00025,
                    peak_threshold_rel=0.1,
                    diagnostic=1,
                    pltshow=1)

    from nubo.fitt import psf_fit
    from nubo.io import psf_annotate

    df, plt_fitt = psf_fit(frames[0], f, 1, 1, 1)
    """

    # """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~Prepare the dataformat~~~~~~~~~~~~~~~~~~~~~~~~~~
    # """
    df = pd.DataFrame([], columns=['frame', 'x_raw', 'y_raw', 'r',
            'A', 'x', 'y', 'sig_x', 'sig_y', 'phi',
            'area', 'mass', 'dist_err', 'sigx_to_sigraw', 'sigy_to_sigraw'])

    df['frame'] = blobs_df['frame']
    df['x_raw'] = blobs_df['x']
    df['y_raw'] = blobs_df['y']
    df['r'] = blobs_df['r']

    # """
    # ~~~~~~~~~~~~~~~~~~~~~~~Fit each blob. If fail, pass~~~~~~~~~~~~~~~~~~~~~~~
    # """

    good_fitting_num = 0
    for i in df.index:
        x0 = int(df.at[i, 'x_raw'])
        y0 = int(df.at[i, 'y_raw'])
        delta = int(round(df.at[i, 'r']))
        Sub_img = pims_frame[x0-delta:x0+delta+1, y0-delta:y0+delta+1]

        try:
            p, p_err = fit_gaussian_2d(Sub_img)
            A = p[0]
            x0_refined = x0 - delta + p[1]
            y0_refined = y0 - delta + p[2]
            sig_x = p[3]
            sig_y = p[4]
            phi = p[5]
            sig_raw = df.at[i, 'sig_raw']
            df.at[i, 'A'] = A
            df.at[i, 'x'] = x0_refined
            df.at[i, 'y'] = y0_refined
            df.at[i, 'sig_x'] = sig_x
            df.at[i, 'sig_y'] = sig_y
            df.at[i, 'phi'] = phi
            df.at[i, 'area'] = np.pi * sig_x * sig_y
            df.at[i, 'mass'] = Sub_img.sum()
            df.at[i, 'dist_err'] = ((x0_refined - x0)**2 + \
                            (y0_refined - y0)**2) ** 0.5
            df.at[i, 'sigx_to_sigraw'] = sig_x / sig_raw
            df.at[i, 'sigy_to_sigraw'] = sig_y / sig_raw

            # """
            # ~~~~~~~~Count the good fitting number with virtual filters~~~~~~~~
            # """

            if (x0_refined - x0)**2 + (y0_refined - y0)**2 \
                    < (diag_max_dist_err)**2 \
            and sig_x < sig_raw * diag_max_sig_to_sigraw \
            and sig_y < sig_raw * diag_max_sig_to_sigraw :
                good_fitting_num = good_fitting_num + 1
        except:
            pass

    print("Predict good fitting number and ratio in frame %d: [%d, %.2f]" %
            (pims_frame.frame_no, good_fitting_num,
            good_fitting_num/len(blobs_df)))

    psf_df = df

    # """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Diagnostic~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # """

    plt_array_2d = []
    if diagnostic:
        import matplotlib.pyplot as plt
        from nubo.io import plt2rgbndarray
        from matplotlib.ticker import FormatStrFormatter
        f1 = df.copy()

        df_filt = pd.DataFrame([], columns=['tot_foci_num'],
                index=['segm', 'fitt', 'dist_err', 'sigx_to_sigraw',
                        'sigy_to_sigraw'])
        df_filt.loc['segm'] = len(f1)
        f1 = f1.dropna(how='any')
        df_filt.loc['fitt'] = len(f1)
        f1 = f1[ f1['dist_err']<diag_max_dist_err ]
        df_filt.loc['dist_err'] = len(f1)
        f1 = f1[ f1['sigx_to_sigraw']<diag_max_sig_to_sigraw ]
        df_filt.loc['sigx_to_sigraw'] = len(f1)
        f1 = f1[ f1['sigy_to_sigraw']<diag_max_sig_to_sigraw ]
        df_filt.loc['sigy_to_sigraw'] = len(f1)
        print(df_filt)

        image = pims_frame
        fig = plt.figure(figsize=(12, 9))

        ax0 = plt.subplot2grid((5,5),(0,0), colspan=5, rowspan=5)
        ax0.imshow(image, cmap="gray", aspect='equal')

        # """
        # ~~~~~~~~~~~~~~~~~~~Add fitting contour to the image~~~~~~~~~~~~~~~~~~~
        # """

        for i in f1.index:
            Fitting_X = np.indices(image.shape)
            p0,p1,p2,p3,p4,p5 = (f1.at[i,'A'], f1.at[i,'x'], f1.at[i,'y'],
                    f1.at[i,'sig_x'], f1.at[i,'sig_y'], f1.at[i,'phi'])
            Fitting_img = gaussian_2d(Fitting_X,p0,p1,p2,p3,p4,p5)
            contour_img = np.zeros(image.shape)
            x1,y1,r1 = f1.at[i,'x'], f1.at[i,'y'], f1.at[i,'r']
            x1 = int(round(x1))
            y1 = int(round(y1))
            r1 = int(round(r1))
            contour_img[x1-r1:x1+r1+1,
                        y1-r1:y1+r1+1] = Fitting_img[x1-r1:x1+r1+1,
                                                     y1-r1:y1+r1+1]
            ax0.contour(contour_img, cmap='cool')

        # """
        # ~~~~~~~~~~~~~~~~~Annotate truth_df, segm_df, psf_df~~~~~~~~~~~~~~~~~~~
        # """

        anno_blob(ax0, f1, marker='x', plot_r=1, color=(1,0,0,0.8))

        if isinstance(segm_df, pd.DataFrame):
            anno_scatter(ax0, segm_df, marker='^', color=(0,0,1,0.8))

        if isinstance(truth_df, pd.DataFrame):
            anno_scatter(ax0, truth_df, marker='o', color=(0,1,0,0.8))

        ax0.text(0.95,
                0.00,
                """
                Predict good fitting focai num and ratio: %d, %.2f
                """ %(good_fitting_num, good_fitting_num/len(blobs_df)),
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize = 12,
                color = (1, 1, 1, 0.8),
                transform=ax0.transAxes)
        plt_array_2d = plot_end(fig, pltshow)

    return psf_df, plt_array_2d


def psf_batch(pims_frames,
            blobs_df,
            diagnostic=False,
            pltshow=False,
            diag_max_dist_err=1,
            diag_max_sig_to_sigraw = 2,
            truth_df=None,
            segm_df=None):
    """
    Point spread function fitting for the whole movie.

    Parameters
    ----------
    pims_frames : pims.Frame object
        Whole movie in the format of pims.Frame.
    bolbs_df : DataFrame
        bolb_df with columns of 'x', 'y', 'r', 'frame'.
    diagnostic : bool, optional
        If true, print the diagnostic strings.
    pltshow : bool, optional
        If true, show diagnostic plot.
    diag_max_dist_err : float, optional
        Virtual max_dist_err filter.
    diag_max_sig_to_sigraw : float, optional
        Virtual diag_max_sig_to_sigraw filter.
    truth_df : DataFrame, optional
        Ground truth DataFrame with columns of 'x', 'y', 'frame'.
    segm_df : DataFrame, optional
        Segmentation DataFrame with columns of 'x', 'y', 'frame'.

    Returns
    -------
    psf_df : DataFrame
        columns=['frame', 'x_raw', 'y_raw', 'r',
                'A', 'x', 'y', 'sig_x', 'sig_y', 'phi','area', 'mass',
                'dist_err', 'sigx_to_sigraw', 'sigy_to_sigraw']
    plt_array_3d :  3d ndarray
        3D ndarray of diagnostic plot.

    Examples
    --------
    import os as _os
    import os.path as osp
    data_dir = osp.abspath(osp.dirname(__file__))
    fname = _os.path.join(data_dir,
               'nubo/data/nucleosome.tif')

    import pims
    from nubo.segm import blob_locate
    frames = pims.open(fname)
    f, plt_segm = blob_locate(frames[0],
                    min_sigma=1,
                    max_sigma=2,
                    num_sigma=5,
                    blob_threshold=0.00025,
                    peak_threshold_rel=0.1,
                    diagnostic=1,
                    pltshow=1)

    from nubo.fitt import psf_fit
    from nubo.io import psf_annotate

    df, plt_fitt = psf_fit(frames[0], f, 1, 1, 1)
    """

    columns=['frame', 'x_raw', 'y_raw', 'r',
            'A', 'x', 'y', 'sig_x', 'sig_y', 'phi',
            'area', 'mass', 'dist_err', 'sigx_to_sigraw', 'sigy_to_sigraw']
    df = pd.DataFrame([], columns=columns)
    plt_array_3d = []
    for i in range(len(pims_frames)):
        current_frame = pims_frames[i]
        fnum = current_frame.frame_no
        current_blob = blobs_df[blobs_df.frame == fnum]

        if isinstance(truth_df, pd.DataFrame):
            curr_truth_df = truth_df[truth_df.frame == fnum]
        else:
            curr_truth_df = None

        if isinstance(segm_df, pd.DataFrame):
            current_segm_df = segm_df[segm_df.frame == fnum]
        else:
            current_segm_df = None

        tmp_psf_df, tmp_plt_array_2d = psf_fit(pims_frame=current_frame,
                       blobs_df=current_blob,
                       diag_max_dist_err=diag_max_dist_err,
                       diag_max_sig_to_sigraw = diag_max_sig_to_sigraw,
                       diagnostic=diagnostic,
                       pltshow=pltshow,
                       truth_df=curr_truth_df,
                       segm_df=current_segm_df)
        df = pd.concat([df, tmp_psf_df], sort=False)
        plt_array_3d.append(tmp_plt_array_2d)

    psf_df = df
    plt_array_3d = np.array(plt_array_3d)

    return psf_df, plt_array_3d

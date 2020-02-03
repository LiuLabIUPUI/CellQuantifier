import math
import numpy as np
import matplotlib as mpl
from matplotlib import patches
import matplotlib.pyplot as plt
import trackpy as tp
from .add_colorbar import add_outside_colorbar
from ._add_scalebar import add_scalebar


def set_ylim_reverse(ax):
    """
    This function is needed for annotation. Since ax.imshow(img) display
    the img in a different manner comparing with traditional axis.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.
    """
    bottom, top = ax.get_ylim()
    if top > bottom:
        ax.set_ylim(top, bottom)

def anno_ellipse(ax, regionprops, linewidth=1, color=(1,0,0,0.8)):
    """
    Annotate ellipse in matplotlib axis.
    The ellipse parameters are obtained from regionprops object of skimage.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.
    regionprops : list of object
        Measure properties of labeled image regions.
        regionprops is the return value of skimage.measure.regionprops().
    linewidth: float, optional
        Linewidth of the ellipse.
    color: tuple, optional
        color of the ellipse.

    Returns
    -------
    Annotate edge, long axis, short axis of ellipses.
    """

    set_ylim_reverse(ax)

    for region in regionprops:
        row, col = region.centroid
        y0, x0 = row, col
        orientation = region.orientation
        ax.plot(x0, y0, '.', markersize=15, color=color)
        x1 = x0 + math.cos(orientation) * 0.5 * region.minor_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * region.minor_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * region.major_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * region.major_axis_length
        ax.plot((x0, x1), (y0, y1), '-', linewidth=linewidth, color=color)
        ax.plot((x0, x2), (y0, y2), '-', linewidth=linewidth, color=color)
        curr_e = patches.Ellipse((x0, y0), width=region.minor_axis_length,
                        height=region.major_axis_length,
                        angle=-orientation/math.pi*180, facecolor='None',
                        linewidth=linewidth, edgecolor=color)
        ax.add_patch(curr_e)

def anno_blob(ax, blob_df, marker='s', plot_r=True, color=(0,1,0,0.8)):
    """
    Annotate blob in matplotlib axis.
    The blob parameters are obtained from blob_df.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.
    blob_df : DataFrame
        bolb_df has columns of 'x', 'y', 'r'.
    makers: string, optional
        The marker for center of the blob.
    plot_r: bool, optional
        If True, plot the circle.
    color: tuple, optional
        Color of the marker and circle.

    Returns
    -------
    Annotate center and the periphery of the blob.
    """

    set_ylim_reverse(ax)

    f = blob_df
    for i in f.index:
        y, x, r = f.at[i, 'x'], f.at[i, 'y'], f.at[i, 'r']
        ax.scatter(x, y,
                    s=10,
                    marker=marker,
                    c=[color])
        if plot_r:
            c = plt.Circle((x,y), r, color=color,
                           linewidth=1, fill=False)
            ax.add_patch(c)

def anno_scatter(ax, scatter_df, marker = 'o', color=(0,1,0,0.8)):
    """
    Annotate scatter in matplotlib axis.
    The scatter parameters are obtained from scatter_df.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.
    scatter_df : DataFrame
        scatter_df has columns of 'x', 'y'.
    makers: string, optional
        The marker for the position of the scatter.
    color: tuple, optional
        Color of the marker and circle.

    Returns
    -------
    Annotate scatter in the ax.
    """

    set_ylim_reverse(ax)

    f = scatter_df
    for i in f.index:
        y, x = f.at[i, 'x'], f.at[i, 'y']
        ax.scatter(x, y,
                    s=10,
                    marker=marker,
                    c=[color])


def anno_traj(ax, df,
            image=np.array([]),
            pixel_size=1,
            scalebar_pos='upper right',
            scalebar_fontsize='large',
            show_traj_num=True,
            fontname='Arial',
            cb_min=None,
            cb_max=None,
            cb_major_ticker=None,
            cb_minor_ticker=None,
            show_particle_label=False,
            choose_particle=None,
            show_colorbar=True):
    """
    Annotate trajectories in matplotlib axis.
    The trajectories parameters are obtained from blob_df.
    The colorbar locates "outside" of the traj figure.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate trajectories.

    df : DataFrame
		DataFrame containing 'D', 'frame', 'x', and 'y' columns

	image: 2D ndarray
		The image the trajectories will be plotted on

    pixel_size: float
		The pixel_size of the images in microns/pixel

    scalebar_pos: string
        string for scalebar position. like 'upper right', 'lower right' ...

    show_traj_num: bool
        If true, show a text with trajectory number

    fontname: string
        Font used in the figure. Default is 'Arial'

    cb_min, cb_max: float
        [cb_min, cb_max] is the color bar range.

    cb_major_ticker, cb_minor_ticker: float
        Major and minor setting for the color bar

    show_particle_label : bool
        If true, add particle label in the figure.

    choose_particle : None or integer
        If particle number specifier, only plot that partcle.

    Returns
    -------
    Annotate trajectories in the ax.
    """
    # """
    # ~~~~~~~~~~~If choose_particle is True, prepare df and image~~~~~~~~~~~~~~
    # """
    original_df = df.copy()
    if choose_particle != None:
        df = df[ df['particle']==choose_particle ]

        r_min = int(round(df['x'].min()))
        c_min = int(round(df['y'].min()))
        delta_x = int(round(df['x'].max()) - round(df['x'].min()))
        delta_y = int(round(df['y'].max()) - round(df['y'].min()))
        delta = int(max(delta_x, delta_y))
        if delta < 1:
            delta = 1
        r_max = r_min + delta_x
        c_max = c_min + delta_y

        df['x'] = df['x'] - r_min
        df['y'] = df['y'] - c_min
        # print('#############################')
        # print(df['x'].min(), df['y'].min())
        # print(df['x'].max(), df['y'].max())
        # print('#############################')

        image = image[r_min:r_max+1, c_min:c_max+1]
    # """
    # ~~~~~~~~~~~Check if df is empty. Plot the image if True~~~~~~~~~~~~~~
    # """
    if df.empty:
    	return

    if image.size != 0:
        ax.imshow(image, cmap='gray', aspect='equal')
        plt.box(False)


    # """
    # ~~~~~~~~~~~Add pixel size scale bar~~~~~~~~~~~~~~
    # """
    add_scalebar(ax, pixel_size=pixel_size, units='um',
                sb_color=(1,1,1),
                sb_pos='upper right',
                length_fraction=0.3,
                height_fraction=0.02,
                box_color=(1,1,1),
                box_alpha=0,
                fontname='Arial',
                fontsize=scalebar_fontsize)


    # """
    # ~~~~~~~~~~~customized the colorbar, then add it~~~~~~~~~~~~~~
    # """
    modified_df, colormap = add_outside_colorbar(ax, original_df,
                        label_font_size='large',
                        cb_min=cb_min,
                        cb_max=cb_max,
                        cb_major_ticker=cb_major_ticker,
                        cb_minor_ticker=cb_minor_ticker,
                        show_colorbar=show_colorbar)


    # """
    # ~~~~~~~~~~~Plot the color coded trajectories using colorbar norm~~~~~~~~~~~~~~
    # """
    if choose_particle:
        df['D_norm'] = modified_df[ modified_df['particle']== choose_particle ]['D_norm']
    else:
        df = modified_df

    ax.set_aspect(1.0)
    particles = df.particle.unique()
    for particle_num in particles:
        traj = df[df.particle == particle_num] # traj = df[df.particle == particles[i]]
        traj = traj.sort_values(by='frame')
        ax.plot(traj['y'], traj['x'], linewidth=1,
        			color=colormap(traj['D_norm'].mean()))
        if show_particle_label:
            ax.text(traj['y'].mean(), traj['x'].mean(),
                    particle_num, color=(0, 1, 0))

    if show_traj_num:
        ax.text(0.95,
                0.00,
                """
                Density Total trajectory number: %d
                """ %(len(particles)),
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize = 12,
                color = (0.5, 0.5, 0.5, 0.5),
                transform=ax.transAxes,
                weight = 'bold',
                fontname = fontname)


    # """
    # ~~~~~~~~~~~Set ax format~~~~~~~~~~~~~~
    # """
    set_ylim_reverse(ax)
    ax.set_xticks([])
    ax.set_yticks([])

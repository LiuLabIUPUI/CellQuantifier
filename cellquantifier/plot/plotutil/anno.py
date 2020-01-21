import math
import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
import trackpy as tp
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

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


def anno_traj(ax, blobs_df, image, pixel_size, frame_rate):
    """
    Annotate trajectories in matplotlib axis.
    The trajectories parameters are obtained from blob_df.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.

    blobs_df : DataFrame
		DataFrame containing 'D', 'frame', 'x', and 'y' columns

	image: 2D ndarray
		The image the trajectories will be plotted on

    pixel_size: float
		The pixel_size of the images in microns/pixel

    frame_rate: float
		Frames per second (fps) of the video

    Returns
    -------
    Annotate trajectories in the ax.
    """

    set_ylim_reverse(ax)

    # """
    # ~~~~~~~~~~~Check if blobs_df is empty~~~~~~~~~~~~~~
    # """
    if blobs_df.empty:
    	return

    # Calculate individual msd
    im = tp.imsd(blobs_df, mpp=pixel_size, fps=frame_rate, max_lagtime=np.inf)

    #Get the diffusion coefficient for each individual particle
    D_ind = blobs_df.drop_duplicates('particle')['D'].mean()

    #Plot the image
    ax.imshow(image, cmap='gray', aspect='equal')
    ax.set_xlim((0, image.shape[1]))
    ax.set_ylim((image.shape[0], 0))

    # """
    # ~~~~~~~~~~~Add D value scale bar to left plot~~~~~~~~~~~~~~
    # """

    scalebar = ScaleBar(pixel_size, 'um', location = 'upper right')
    ax.add_artist(scalebar)

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.1)
    norm = mpl.colors.Normalize(vmin = blobs_df['D'].min(), vmax = blobs_df['D'].max())
    cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=mpl.cm.jet, norm=norm, orientation='vertical')
    cb1.set_label(r'$\mathbf{D (nm^{2}/s)}$')
    fig = plt.gcf()
    fig.add_axes(ax_cb)

    colormap = plt.cm.get_cmap('jet')
    blobs_df['D_norm'] = blobs_df['D']/(blobs_df['D'].max()) #normalize D column to maximum D value

    # """
    # ~~~~~~~~~~~Plot the color coded trajectories~~~~~~~~~~~~~~
    # """
    particles = blobs_df.particle.unique()
    for i in range(len(particles)):
    	traj = blobs_df[blobs_df.particle == particles[i]]
    	traj = traj.sort_values(by='frame')
    	ax.plot(traj.y, traj.x, linewidth=1,
    				color=colormap(traj['D_norm'].mean()))

    ax.set_aspect(1.0)

    ax.text(0.95,
            0.00,
            """
            Total trajectory number: %d
            """ %(len(particles)),
            horizontalalignment='right',
            verticalalignment='bottom',
            fontsize = 12,
            color = (0.5, 0.5, 0.5, 0.5),
            transform=ax.transAxes)

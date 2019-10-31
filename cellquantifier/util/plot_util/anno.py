import math
from matplotlib import patches
import matplotlib.pyplot as plt

def anno_ellipse(ax, regionprops, linewidth=2.5, color=(1,0,0,0.8)):
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
    The scatter parameters are obtained from blob_df.

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

    f = scatter_df
    for i in f.index:
        y, x = f.at[i, 'x'], f.at[i, 'y']
        ax.scatter(x, y,
                    s=10,
                    marker=marker,
                    c=[color])

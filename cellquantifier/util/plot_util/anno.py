import math
from matplotlib import patches

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

    Returns
    -------
    Annotate edge, long axis, short axis of ellipses.
    """
    for region in regionprops:
        y0, x0 = region.centroid
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

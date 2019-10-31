import matplotlib.pyplot as plt
import numpy as np

def plt2array(fig):
    """
    Save matplotlib.pyplot figure to numpy rgbndarray.

    Parameters
    ----------
    fig : object
        matplotlib figure object.

    Returns
    -------
    rgb_array_3d: ndarray
        3d ndarray represting the figure

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from cellquantifier.io._plt2array import plt2array
    >>> t = np.linspace(0, 4*np.pi, 1000)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(t, np.cos(t))
    >>> ax.plot(t, np.sin(t))
    >>> result_array_3d = plt2array(fig)
    >>> print(result_array_3d.shape)
    (480, 640, 3)
    """

    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    rgb_array_3d = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

    return rgb_array_3d

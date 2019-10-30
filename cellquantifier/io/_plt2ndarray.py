def plt2rgbndarray(fig):

    """
    save matplotlib.pyplot figure to numpy rgbndarray

    test code
    ------------------------------------------------
    import matplotlib.pyplot as plt
    import numpy as np
    from nubo.io import plt2rgbndarray

    t = np.linspace(0, 4*np.pi, 1000)
    fig, ax = plt.subplots()
    ax.plot(t, np.cos(t))
    ax.plot(t, np.sin(t))
    out = plt2rgbndarray(fig)
    print(out)
    ------------------------------------------------
    """

    """
    Save matplotlib.pyplot figure to numpy rgbndarray.

    Parameters
    ----------
    fig : object
        matplotlib figure object.

    Returns
    -------
    rgb_array_ed: ndarray
    
    Annotate edge, long axis, short axis of ellipses.

    Examples
    --------
    >>> from skimage import data, feature
    >>> feature.blob_dog(data.coins(), threshold=.5, max_sigma=40)
    array([[267.      , 359.      ,  16.777216],
           [267.      , 115.      ,  10.48576 ],
           [263.      , 302.      ,  16.777216],
           [263.      , 245.      ,  16.777216],
           [261.      , 173.      ,  16.777216],
           [260.      ,  46.      ,  16.777216],
           [198.      , 155.      ,  10.48576 ],
           [196.      ,  43.      ,  10.48576 ],
           [195.      , 102.      ,  16.777216],
           [194.      , 277.      ,  16.777216],
           [193.      , 213.      ,  16.777216],
           [185.      , 347.      ,  16.777216],
           [128.      , 154.      ,  10.48576 ],
           [127.      , 102.      ,  10.48576 ],
           [125.      , 208.      ,  10.48576 ],
           [125.      ,  45.      ,  16.777216],
           [124.      , 337.      ,  10.48576 ],
           [120.      , 272.      ,  16.777216],
           [ 58.      , 100.      ,  10.48576 ],
           [ 54.      , 276.      ,  10.48576 ],
           [ 54.      ,  42.      ,  16.777216],
           [ 52.      , 216.      ,  16.777216],
           [ 52.      , 155.      ,  16.777216],
           [ 45.      , 336.      ,  16.777216]])
    """

    import matplotlib.pyplot as plt
    import numpy as np

    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    rgbndarray = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

    return rgbndarray

import os.path as path

data_dir = path.abspath(path.dirname(__file__))

def simulated_cell():

    """Gray-level "simulated_cell" image.
    Average diffusion coefficient for simulatd particles is 0.2
    Often used for segmentation and particle tracking examples.

    Returns
    -------
    simulated_cell : (800, 800) uint8 2D ndarray
        Simulated cell image.
    """

    return _load("simulated_cell.tif")

def _load(f, as_gray=False):

    """Load an image file located in the data directory.

    Parameters
    ----------
    f : string
        File name.
    as_gray : bool, optional
        Whether to convert the image to grayscale.

    Returns
    -------
    img : ndarray
        Image loaded from data directory

    """
    import pims
    file = pims.open(path.join(data_dir, f))
    
    return file

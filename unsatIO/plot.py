import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from unsatIO.io import slicer


def plot_slice(array, index=1, axis=0, *args, **kwargs):
    """Plot a slice as a 2D image

    Args:
        array: a numpy array
        index: the position of the slice
        axis: the axis of the slice (0 for horizontal, 1 and 2 for vertical)

    Returns:
        An AxesImage object
    """
    sliced = slicer(array, index, axis)
    return plt.imshow(sliced, *args, **kwargs)

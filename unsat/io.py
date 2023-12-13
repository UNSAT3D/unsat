from PIL import Image
import numpy as np


def tif_to_numpy(tif_file):
    """Loads a tif image as a numpy array

    Args:
        tif_file: path to the tif image

    Returns:
        A three-dimensional numpy array
    """
    images = []
    with Image.open(tif_file) as img:
        i = 0
        while True:
            try:
                img.seek(i)
                images.append(np.array(img))
                i += 1
            except EOFError:
                # Reached end of image sequence (file)
                break

    images = np.array(images)
    return images


def slicer(array, index, axis=0):
    """Slice the array along the main axes

    Args:
        array: a numpy array
        index: the position of the slice
        axis: the axis of the slice (0 for horizontal, 1 and 2 for vertical)

    Returns:
        A two-dimensional numpy array
    """
    if axis == 0:
        sliced = array[index, :, :]
    elif axis == 1:
        sliced = array[:, index, :]
    elif axis == 2:
        sliced = array[:, :, index]
    else:
        raise ValueError("axis must be 0, 1 or 2")

    return sliced


def center(array):
    """Returns the coordinates of the center of an array"""
    s = array.shape
    return list(np.floor(coord / 2) for coord in s)


def radius(circular_slice):
    """Estimates the radius (in pixels) of a circular slice"""
    c = center(circular_slice)
    s = circular_slice.shape
    return s[0] - c[0]

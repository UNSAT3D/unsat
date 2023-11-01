from PIL import Image
import numpy as np

def tif_to_numpy(tif_file):
    """Loads a tif image as a numpy array
    
    Args:
        tif_file: path to the tif image
    
    Returns:
        A numpy array
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

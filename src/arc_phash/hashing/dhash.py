"""
    Compute dhash
"""

import dhash
import numpy as np
from PIL import Image


def dhash_img(img: Image) -> np.ndarray:
    """Generate the dhash of an image

    Args:
        img: image to hash

    Returns:
        numpy array of binary hash
    """
    row, col = dhash.dhash_row_col(img, size=8)
    return np.array(list(format(row, "016b") + format(col, "016b")), dtype=np.float32)

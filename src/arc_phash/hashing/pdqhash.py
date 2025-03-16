"""
    Compute pdq hash on an image
"""

import numpy as np
import pdqhash
from PIL import Image


def pdqhash_img(img: Image) -> np.ndarray:
    """Generate a pdqhash of an image

    Args:
        img: image to hash

    Returns:
        np array of binary hash
    """
    hash, _ = pdqhash.compute(np.asarray(img))
    return np.asarray(hash, dtype=np.float32)

"""
    Utility functions for image manipulation
"""


def get_resize_tuple(pil_image, target_size=1024):
    """Calculate resize dimensions based on aspect ratio

    Args:
        pil_image: input image
        target_size: target resize. Defaults to 1024.
    """
    if pil_image.size[0] < pil_image.size[1]:
        # Portrait orientation
        resize_tuple = (
            target_size,
            int(target_size * pil_image.size[1] / pil_image.size[0]),
        )
    else:
        # Landscape orientation
        resize_tuple = (
            int(target_size * pil_image.size[0] / pil_image.size[1]),
            target_size,
        )

    # check it is divisible by 8 and return
    return (
        resize_tuple[0] - resize_tuple[0] % 8,
        resize_tuple[1] - resize_tuple[1] % 8,
    )


def centre_crop(image, new_width, new_height):
    """Perform a centre crop on an input image

    Args:
        image: input image
        new_width: new width to crop to
        new_height: new height to crop to
    """
    width, height = image.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return image.crop((left, top, right, bottom))


def resize_and_crop(pil_image, target_size=1024):
    """Resize and crop and image for purposes of AI manipulation

    Args:
        pil_image: input image
        target_size: target resize. Defaults to 1024.
    """
    resize_tuple = get_resize_tuple(pil_image, target_size=target_size)
    pil_image = pil_image.resize(resize_tuple)
    return centre_crop(pil_image, target_size, target_size)

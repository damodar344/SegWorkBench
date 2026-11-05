"""General image processing utilities."""

import numpy as np
from PIL import Image, ImageFilter


def crop_to_content(
    image_pil: Image.Image, mask_pil: Image.Image
) -> tuple[Image.Image, Image.Image]:
    """Crop *image_pil* and *mask_pil* to the bounding box of mask foreground.

    A small median filter is applied to the mask before computing the bbox.
    """
    mask = mask_pil.convert("L").filter(ImageFilter.MedianFilter(3))
    arr = np.array(mask)
    ys, xs = np.where(arr >= 128)
    if ys.size == 0 or xs.size == 0:
        return image_pil, mask_pil

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    pad = 2

    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(image_pil.width - 1, x1 + pad)
    y1 = min(image_pil.height - 1, y1 + pad)

    box = (x0, y0, x1 + 1, y1 + 1)
    return image_pil.crop(box), mask_pil.crop(box)

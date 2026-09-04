"""Shared image crop/resize preprocessing.

Used by data-collection preview, training (dataset.py), and inference
(deploy_flowbot_w_policy.py, debug/visualize/extract scripts) so all four
stages apply the exact same transform to camera frames. Anchor-based crop:
crop_x/crop_y in [0, 1] pick where the crop window sits in the source frame
(0=left/top, 0.5=center, 1=right/bottom) instead of always cropping from the
center.
"""

import cv2


def crop_and_resize(img, image_size, crop_scale=1.5, crop_x=0.5, crop_y=0.5):
    """Crop a window anchored at (crop_x, crop_y) and resize to image_size.

    Args:
        img: (H, W, C) array
        image_size: (target_h, target_w)
        crop_scale: crop window size as a multiple of image_size
        crop_x, crop_y: crop anchor in [0, 1] (0=left/top, 0.5=center, 1=right/bottom)

    Returns:
        (target_h, target_w, C) array, same dtype as img
    """
    h, w = img.shape[:2]
    target_h, target_w = image_size

    crop_h = min(h, int(target_h * crop_scale))
    crop_w = min(w, int(target_w * crop_scale))
    start_h = int((h - crop_h) * crop_y)
    start_w = int((w - crop_w) * crop_x)
    img_cropped = img[start_h:start_h + crop_h, start_w:start_w + crop_w]

    return cv2.resize(img_cropped, (target_w, target_h))

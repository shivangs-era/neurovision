"""
Image preprocessing, normalization, bounding box extraction, and center-of-mass centering.
Mirrors the authentic MNIST standardization pipeline to ensure optimal inference on drawn inputs.
"""

from typing import Union, Tuple
import numpy as np
from PIL import Image
import torch


def preprocess_drawn_image(
    image_input: Union[Image.Image, np.ndarray],
    invert: bool = True,
    center_mass: bool = True,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Standardizes arbitrary drawn or uploaded imagery into a normalized 28x28 tensor.

    Steps:
      1. Converts to grayscale.
      2. Handles transparency / background polarity.
      3. Extracts bounding box of active pixels.
      4. Aspect-ratio-preserving resize into a 20x20 bounding box.
      5. Embeds in 28x28 canvas with center-of-mass translation.
      6. Converts to [1, 1, 28, 28] PyTorch float tensor in [0, 1].
    """
    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 3 and image_input.shape[2] == 4:
            # RGBA format: extract alpha channel or RGB
            alpha = image_input[:, :, 3]
            if np.max(alpha) > 0 and np.min(alpha) < 255:
                # Drawing has alpha strokes
                rgb = image_input[:, :, :3]
                gray = np.mean(rgb, axis=2)
                # If background is transparent, mask using alpha
                img_gray = (gray * (alpha / 255.0)).astype(np.uint8)
            else:
                img_gray = np.mean(image_input[:, :, :3], axis=2).astype(np.uint8)
        elif image_input.ndim == 3 and image_input.shape[2] == 3:
            img_gray = np.mean(image_input, axis=2).astype(np.uint8)
        else:
            img_gray = image_input.astype(np.uint8)
        pil_img = Image.fromarray(img_gray)
    else:
        pil_img = image_input.convert("L")

    img_arr = np.array(pil_img, dtype=np.float32)

    # Detect polarity: if background is light (>128 mean) and strokes are dark, invert so strokes are white on black
    if invert and np.mean(img_arr) > 127.5:
        img_arr = 255.0 - img_arr

    # Threshold very faint background noise
    threshold = 30.0
    img_arr[img_arr < threshold] = 0.0

    # If completely blank, return empty 28x28
    if np.max(img_arr) == 0.0:
        empty_arr = np.zeros((28, 28), dtype=np.float32)
        tensor = torch.tensor(empty_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor, empty_arr

    # Find active bounding box
    rows = np.any(img_arr > 0, axis=1)
    cols = np.any(img_arr > 0, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop active region
    cropped = img_arr[rmin : rmax + 1, cmin : cmax + 1]
    crop_h, crop_w = cropped.shape

    # Fit into 20x20 while preserving aspect ratio
    if crop_h > crop_w:
        new_h = 20
        new_w = max(1, int(round((crop_w / crop_h) * 20.0)))
    else:
        new_w = 20
        new_h = max(1, int(round((crop_h / crop_w) * 20.0)))

    crop_pil = Image.fromarray(cropped.astype(np.uint8)).resize(
        (new_w, new_h), resample=Image.Resampling.BILINEAR
    )
    scaled_arr = np.array(crop_pil, dtype=np.float32)

    # Place in 28x28 blank canvas centered geometrically first
    canvas = np.zeros((28, 28), dtype=np.float32)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    canvas[start_y : start_y + new_h, start_x : start_x + new_w] = scaled_arr

    # Translate by center of mass to match MNIST benchmark
    if center_mass:
        total_mass = np.sum(canvas)
        if total_mass > 0:
            cy, cx = np.indices(canvas.shape)
            com_y = np.sum(cy * canvas) / total_mass
            com_x = np.sum(cx * canvas) / total_mass
            # Shift canvas so (com_y, com_x) aligns with (14.0, 14.0)
            shift_y = int(round(14.0 - com_y))
            shift_x = int(round(14.0 - com_x))

            # Apply roll with edge clipping
            canvas = np.roll(canvas, shift_y, axis=0)
            canvas = np.roll(canvas, shift_x, axis=1)

            # Clear rolled borders
            if shift_y > 0:
                canvas[:shift_y, :] = 0.0
            elif shift_y < 0:
                canvas[shift_y:, :] = 0.0

            if shift_x > 0:
                canvas[:, :shift_x] = 0.0
            elif shift_x < 0:
                canvas[:, shift_x:] = 0.0

    # Normalize to [0.0, 1.0]
    canvas = canvas / 255.0
    canvas = np.clip(canvas, 0.0, 1.0)

    tensor = torch.tensor(canvas, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor, canvas

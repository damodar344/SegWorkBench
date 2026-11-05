"""Inference engine for UNet segmentation."""

import logging
import os
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from segworkbench.models.unet import UNet
from segworkbench.utils.image_utils import crop_to_content

logger = logging.getLogger(__name__)


def _auto_device(device: Optional[torch.device] = None) -> torch.device:
    """Pick the best available accelerator."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except AttributeError:
        pass
    return torch.device("cpu")


class Inferencer:
    """UNet batch inference runner."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = _auto_device(device)

    @staticmethod
    def _supported_files(directory: str):
        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        return [f for f in os.listdir(directory) if f.lower().endswith(exts)]

    def run(
        self,
        model_path: str,
        input_dir: str,
        output_dir: str,
        out_ext: str = ".png",
        image_size: int = 256,
        threshold: float = 0.5,
        preserve_depth: bool = False,
        should_stop: Optional[Callable[[], bool]] = None,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict:
        """Run inference on all images in *input_dir*.

        Args:
            model_path: Path to a ``.pth`` model checkpoint.
            input_dir: Folder containing input images.
            output_dir: Folder to write results.
            out_ext: Output extension (``.png`` or ``.tif``).
            image_size: Spatial square size fed to the model.
            threshold: Probability threshold for binarising the mask.
            preserve_depth: If ``True``, handle 16-bit / grayscale images
                without forcing 8-bit RGB conversion.
            should_stop: Callable that returns ``True`` when inference should abort.
            on_progress: Callable(current_index, total, filename).

        Returns:
            Dictionary with ``duration_seconds`` and ``output_dir``.
        """
        if should_stop is None:
            should_stop = lambda: False

        model = UNet(n_channels=3, n_classes=1).to(self.device)
        sd = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(sd)
        model.eval()

        rgb_tfm = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        files = self._supported_files(input_dir)
        if not files:
            raise ValueError("No supported images found in input folder.")

        os.makedirs(output_dir, exist_ok=True)
        logger.info(
            "Starting inference on %d images → %s (size=%d, thr=%.2f)",
            len(files), output_dir, image_size, threshold,
        )

        start = time.time()
        for i, fname in enumerate(files, 1):
            if should_stop():
                logger.info("Inference stopped by user at %d/%d", i - 1, len(files))
                break

            fpath = os.path.join(input_dir, fname)
            img = Image.open(fpath)
            orig_w, orig_h = img.size

            if preserve_depth and img.mode in ("L", "I", "I;16", "F"):
                # High-bit-depth / grayscale pipeline
                img_r = T.Resize((image_size, image_size))(img)
                arr = np.array(img_r, dtype=np.float32)
                if img.mode == "I;16":
                    arr = arr / 65535.0
                elif img.mode == "I":
                    arr = arr / 255.0
                elif img.mode == "F":
                    a_min, a_max = arr.min(), arr.max()
                    if a_max > a_min:
                        arr = (arr - a_min) / (a_max - a_min)
                ten = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
                ten = ten.repeat(1, 3, 1, 1)
                ten = (ten - 0.5) / 0.5
            else:
                rgb = img.convert("RGB")
                ten = rgb_tfm(rgb)

            ten = ten.unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = model(ten)
                prob_small = torch.sigmoid(out).cpu()[0, 0]
                prob = F.interpolate(
                    prob_small.unsqueeze(0).unsqueeze(0),
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].numpy()

            mask_arr = (prob > threshold).astype("uint8") * 255
            mask_pil = Image.fromarray(mask_arr, mode="L")

            cropped_img, cropped_mask = crop_to_content(img, mask_pil)

            base = os.path.splitext(fname)[0]
            if out_ext.lower() == ".png":
                rgba = cropped_img.convert("RGBA")
                a = np.array(cropped_mask) / 255.0
                rgb_arr = np.array(rgba, dtype=np.float32)
                rgb_arr[..., 3] = (a * 255.0).astype(np.uint8)
                out_img = Image.fromarray(rgb_arr.astype(np.uint8), "RGBA")
                out_img.save(os.path.join(output_dir, f"segmented_{base}.png"))
            else:
                cropped_mask.save(os.path.join(output_dir, f"mask_{base}.tif"))
                cropped_img.save(os.path.join(output_dir, f"cropped_{base}.tif"))

            if on_progress is not None:
                on_progress(i, len(files), fname)

        duration = time.time() - start
        logger.info("Inference finished in %.1fs → %s", duration, output_dir)
        return {"duration_seconds": duration, "output_dir": output_dir}

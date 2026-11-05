"""Gaussian Mixture Model based image segmentation."""

from typing import Optional

import numpy as np
from PIL import Image

try:
    from sklearn.mixture import GaussianMixture

    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False  # type: ignore[assignment]

try:
    from skimage import filters

    SKIMAGE_OK = True
except Exception:
    SKIMAGE_OK = False  # type: ignore[assignment]


class GMMSegmenter:
    """Hold state and run GMM / threshold-based segmentation on a single image."""

    def __init__(self):
        if not SKLEARN_OK:
            raise ImportError("scikit-learn is required for GMMSegmenter")
        if not SKIMAGE_OK:
            raise ImportError("scikit-image is required for GMMSegmenter")

        self.img: Optional[Image.Image] = None
        self.img_arr: Optional[np.ndarray] = None
        self.non_black: Optional[np.ndarray] = None
        self.gmm: Optional[GaussianMixture] = None
        self.threshold: float = 0.0

    def load_image(self, path: str) -> None:
        """Load a grayscale image and prepare pixel data."""
        img = Image.open(path)
        if img.mode not in ["L", "I", "F"]:
            img = img.convert("L")
        self.img = img
        self.img_arr = np.array(self.img)

        flat = self.img_arr.flatten()
        self.non_black = flat[flat > 0]
        self.gmm = None
        self.threshold = 0.0

    def fit_gmm(self, n_components: int = 2) -> None:
        """Fit a Gaussian Mixture Model to non-black pixels."""
        if self.non_black is None or self.non_black.size == 0:
            raise ValueError("No image loaded or image is completely black.")
        try:
            self.gmm = GaussianMixture(n_components=n_components, random_state=42)
            self.gmm.fit(self.non_black.reshape(-1, 1))
        except Exception as exc:
            raise RuntimeError(f"Failed to fit GMM: {exc}") from exc

    def compute_threshold(self, method: str = "GMM") -> float:
        """Compute a threshold using the chosen method.

        Args:
            method: One of ``GMM``, ``Otsu``, ``Triangle``, ``IsoData``.

        Returns:
            The computed threshold value.
        """
        if self.non_black is None or self.non_black.size == 0:
            raise ValueError("No image data available.")

        if method == "GMM" and self.gmm is not None and self.gmm.n_components >= 2:
            means = self.gmm.means_.flatten()
            covs = self.gmm.covariances_.reshape(-1)
            weights = self.gmm.weights_.flatten()

            order = np.argsort(means)
            i1, i2 = int(order[0]), int(order[1])
            m1, m2 = float(means[i1]), float(means[i2])
            s1, s2 = float(np.sqrt(covs[i1])), float(np.sqrt(covs[i2]))
            w1, w2 = float(weights[i1]), float(weights[i2])

            x = np.linspace(min(m1, m2), max(m1, m2), 5000)
            pdf1 = w1 * np.exp(-0.5 * ((x - m1) / s1) ** 2) / (s1 * np.sqrt(2 * np.pi))
            pdf2 = w2 * np.exp(-0.5 * ((x - m2) / s2) ** 2) / (s2 * np.sqrt(2 * np.pi))
            idx = int(np.argmin(np.abs(pdf1 - pdf2)))
            th = float(x[idx])

        elif method == "Otsu":
            th = float(filters.threshold_otsu(self.non_black))
        elif method == "Triangle":
            th = float(filters.threshold_triangle(self.non_black))
        elif method == "IsoData":
            th = float(filters.threshold_isodata(self.non_black))
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        self.threshold = th
        return th

    def apply_segmentation(self, threshold: Optional[float] = None) -> np.ndarray:
        """Return a binary mask (0 / 255) using *threshold* or the last computed one."""
        if self.img_arr is None:
            raise ValueError("No image loaded.")
        th = threshold if threshold is not None else self.threshold
        mask = (self.img_arr > th).astype(np.uint8) * 255
        return mask

    def get_histogram_data(self, bins: int = 256):
        """Return histogram counts and bin edges for the non-black pixels."""
        if self.non_black is None or self.non_black.size == 0:
            return None, None
        counts, edges = np.histogram(self.non_black, bins=bins)
        return counts, edges

    def get_gmm_curves(self, n_points: int = 800):
        """Return x and per-component pdf values for plotting.

        Returns:
            Tuple of ``(xs, list_of_pdfs)`` or ``(None, None)`` if GMM is not fitted.
        """
        if self.gmm is None or self.non_black is None or self.non_black.size == 0:
            return None, None

        xs = np.linspace(
            self.non_black.min(), self.non_black.max(), n_points
        ).reshape(-1, 1)
        lp = self.gmm.score_samples(xs)
        pdf = np.exp(lp)
        probs = self.gmm.predict_proba(xs)

        curves = []
        for i in range(self.gmm.n_components):
            comp_pdf = probs[:, i] * pdf
            curves.append(comp_pdf)
        return xs.flatten(), curves

"""Tab 3 — GMM Segmentation."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from segworkbench.analysis import GMMSegmenter
from segworkbench.gui.base_tab import BaseTab
from segworkbench.gui.widgets import MplCanvas


SLIDER_STEPS = 10000


class GMMTab(BaseTab):
    """Gaussian-mixture / threshold segmentation tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.segmenter = GMMSegmenter()

        self._img_min: float = 0.0
        self._img_max: float = 255.0
        self._current_threshold: float = 0.0

        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Image path -------------------------------------------------
        row_path = QHBoxLayout()
        self._edit_path = QLineEdit()
        self._edit_path.setReadOnly(True)
        self._edit_path.setPlaceholderText("Image path...")
        row_path.addWidget(self._edit_path, stretch=1)

        btn_open = QPushButton("Open Image")
        btn_open.clicked.connect(self._open_image)
        row_path.addWidget(btn_open)
        layout.addLayout(row_path)

        # --- Options: Gaussians + Method --------------------------------
        row_opts = QHBoxLayout()

        row_opts.addWidget(QLabel("Gaussians:"))
        self._spin_components = QSpinBox()
        self._spin_components.setRange(1, 9)
        self._spin_components.setValue(2)
        self._spin_components.valueChanged.connect(self._refit_gmm)
        row_opts.addWidget(self._spin_components)

        row_opts.addSpacing(20)
        row_opts.addWidget(QLabel("Method:"))

        self._btn_group = QButtonGroup(self)
        for method in ["GMM", "Otsu", "Triangle", "IsoData"]:
            rb = QRadioButton(method)
            rb.setChecked(method == "GMM")
            self._btn_group.addButton(rb)
            row_opts.addWidget(rb)
        self._btn_group.buttonClicked.connect(self._compute_threshold)

        row_opts.addStretch()
        layout.addLayout(row_opts)

        # --- Threshold slider ---------------------------------------------
        row_slider = QHBoxLayout()
        row_slider.addWidget(QLabel("Threshold:"))

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, SLIDER_STEPS)
        self._slider.setValue(0)
        self._slider.valueChanged.connect(self._on_slider_moved)
        row_slider.addWidget(self._slider, stretch=1)

        self._lbl_threshold = QLabel("0.0")
        row_slider.addWidget(self._lbl_threshold)
        layout.addLayout(row_slider)

        # --- Apply button ------------------------------------------------
        btn_apply = QPushButton("Apply Segmentation")
        btn_apply.clicked.connect(self._apply_segmentation)
        layout.addWidget(btn_apply)

        # --- Canvases -----------------------------------------------------
        row_canvases = QHBoxLayout()
        self._canvas_img = MplCanvas(width=4, height=4, dpi=100)
        self._canvas_img.axes.axis("off")
        row_canvases.addWidget(self._canvas_img)

        self._canvas_hist = MplCanvas(width=4, height=4, dpi=100)
        row_canvases.addWidget(self._canvas_hist)
        layout.addLayout(row_canvases, stretch=1)

    # ------------------------------------------------------------------ #
    # File / image loading
    # ------------------------------------------------------------------ #
    def _open_image(self):
        p = self.ask_open_file(
            "Open Image",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.gif);;All files (*.*)",
        )
        if not p:
            return
        self._edit_path.setText(p)

        try:
            self.segmenter.load_image(p)
        except Exception as exc:
            self.show_error("GMM", f"Failed to load image: {exc}")
            return

        if self.segmenter.non_black is None or self.segmenter.non_black.size == 0:
            self.show_error("GMM", "The image appears to be completely black.")
            return

        # Set range based on actual pixel distribution
        self._img_min = float(self.segmenter.non_black.min())
        self._img_max = float(self.segmenter.non_black.max())

        # Show original image and initialise GMM
        self._show_image(self.segmenter.img_arr)
        self._refit_gmm(self._spin_components.value())

    # ------------------------------------------------------------------ #
    # GMM / threshold logic
    # ------------------------------------------------------------------ #
    def _refit_gmm(self, n_components: int):
        """Re-fit GMM when the user changes the number of components."""
        if self.segmenter.non_black is None or self.segmenter.non_black.size == 0:
            return
        try:
            self.segmenter.fit_gmm(n_components)
        except Exception as exc:
            self.show_error("GMM", f"Failed to fit GMM: {exc}")
            return
        self._compute_threshold()

    def _compute_threshold(self, _clicked_button=None):
        """Compute threshold for the currently selected method."""
        if self.segmenter.non_black is None or self.segmenter.non_black.size == 0:
            return

        # Determine active method
        method = "GMM"
        if _clicked_button is not None:
            method = _clicked_button.text()
        else:
            for rb in self._btn_group.buttons():
                if rb.isChecked():
                    method = rb.text()
                    break

        try:
            th = self.segmenter.compute_threshold(method)
        except Exception as exc:
            self.show_error("Threshold", f"Failed to compute threshold: {exc}")
            return

        self._current_threshold = th
        self._lbl_threshold.setText(f"{th:.1f}")

        # Move slider to the new threshold without firing the callback
        slider_pos = self._value_to_slider(th)
        self._slider.blockSignals(True)
        self._slider.setValue(slider_pos)
        self._slider.blockSignals(False)

        self._draw_histogram()

    def _on_slider_moved(self, position: int):
        """User dragged the threshold slider manually."""
        th = self._slider_to_value(position)
        self._current_threshold = th
        self._lbl_threshold.setText(f"{th:.1f}")
        self._draw_histogram()

    # ------------------------------------------------------------------ #
    # Segmentation & display
    # ------------------------------------------------------------------ #
    def _apply_segmentation(self):
        if self.segmenter.img_arr is None:
            self.show_error("GMM", "Load an image first.")
            return
        mask = self.segmenter.apply_segmentation(self._current_threshold)
        self._show_image(mask)

    def _show_image(self, arr: np.ndarray):
        self._canvas_img.axes.clear()
        self._canvas_img.axes.imshow(arr, cmap="gray")
        self._canvas_img.axes.axis("off")
        self._canvas_img.draw_idle()

    # ------------------------------------------------------------------ #
    # Histogram (right canvas)
    # ------------------------------------------------------------------ #
    def _draw_histogram(self):
        ax = self._canvas_hist.axes
        ax.clear()

        if self.segmenter.non_black is None or self.segmenter.non_black.size == 0:
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Frequency")
            self._canvas_hist.draw_idle()
            return

        bins = 256
        counts, _ = self.segmenter.get_histogram_data(bins)
        ax.hist(
            self.segmenter.non_black,
            bins=bins,
            color="black",
            edgecolor="black",
            alpha=0.6,
        )

        # Plot GMM component curves if GMM is selected and fitted
        if self._is_gmm_method() and self.segmenter.gmm is not None:
            xs, curves = self.segmenter.get_gmm_curves(n_points=800)
            if xs is not None and counts is not None:
                hist_max = max(1, int(np.max(counts)))
                for comp_pdf in curves:
                    if comp_pdf.max() > 0:
                        ax.plot(
                            xs,
                            (comp_pdf / comp_pdf.max()) * hist_max,
                            linewidth=2,
                        )

        # Threshold line
        ax.axvline(self._current_threshold, color="green", linestyle="--")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        self._canvas_hist.draw_idle()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _is_gmm_method(self) -> bool:
        for rb in self._btn_group.buttons():
            if rb.isChecked():
                return rb.text() == "GMM"
        return False

    def _slider_to_value(self, position: int) -> float:
        if self._img_max == self._img_min:
            return self._img_min
        t = position / SLIDER_STEPS
        return self._img_min + t * (self._img_max - self._img_min)

    def _value_to_slider(self, value: float) -> int:
        if self._img_max == self._img_min:
            return 0
        t = (value - self._img_min) / (self._img_max - self._img_min)
        return int(round(t * SLIDER_STEPS))

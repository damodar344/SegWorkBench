"""Tab 2 — Batch Inference."""

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from segworkbench.engine import Inferencer
from segworkbench.gui.base_tab import BaseTab
from segworkbench.gui.workers import InferenceWorker


class InferenceTab(BaseTab):
    """Batch-inference tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.inferencer = Inferencer()

        self._worker: InferenceWorker | None = None

        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Paths -------------------------------------------------------
        paths_group = QGroupBox("Paths")
        paths_layout = QVBoxLayout(paths_group)

        row_model = QHBoxLayout()
        row_model.addWidget(QLabel("Model (.pth):"))
        self._edit_model = QLineEdit()
        row_model.addWidget(self._edit_model, stretch=1)
        btn_browse_model = QPushButton("Browse")
        btn_browse_model.clicked.connect(self._browse_model)
        row_model.addWidget(btn_browse_model)
        paths_layout.addLayout(row_model)

        row_in = QHBoxLayout()
        row_in.addWidget(QLabel("Input Folder:"))
        self._edit_input = QLineEdit()
        row_in.addWidget(self._edit_input, stretch=1)
        btn_browse_in = QPushButton("Browse")
        btn_browse_in.clicked.connect(self._browse_input)
        row_in.addWidget(btn_browse_in)
        paths_layout.addLayout(row_in)

        row_out = QHBoxLayout()
        row_out.addWidget(QLabel("Output Folder:"))
        self._edit_output = QLineEdit()
        row_out.addWidget(self._edit_output, stretch=1)
        btn_browse_out = QPushButton("Browse")
        btn_browse_out.clicked.connect(self._browse_output)
        row_out.addWidget(btn_browse_out)
        paths_layout.addLayout(row_out)

        layout.addWidget(paths_group)

        # --- Options -----------------------------------------------------
        opts_group = QGroupBox("Options")
        opts_layout = QHBoxLayout(opts_group)

        opts_layout.addWidget(QLabel("Save as:"))
        self._combo_ext = QComboBox()
        self._combo_ext.addItems([".png", ".tif"])
        opts_layout.addWidget(self._combo_ext)

        opts_layout.addWidget(QLabel("Image size:"))
        self._spin_size = QSpinBox()
        self._spin_size.setRange(64, 2048)
        self._spin_size.setValue(256)
        self._spin_size.setToolTip(
            "Square size fed to the model (original resolution is restored in output)"
        )
        opts_layout.addWidget(self._spin_size)

        opts_layout.addWidget(QLabel("Threshold:"))
        self._spin_threshold = QDoubleSpinBox()
        self._spin_threshold.setRange(0.0, 1.0)
        self._spin_threshold.setDecimals(2)
        self._spin_threshold.setSingleStep(0.05)
        self._spin_threshold.setValue(0.5)
        opts_layout.addWidget(self._spin_threshold)

        self._chk_preserve = QCheckBox("Preserve 16-bit depth")
        self._chk_preserve.setToolTip(
            "Avoid forcing 8-bit RGB conversion for grayscale / 16-bit images"
        )
        opts_layout.addWidget(self._chk_preserve)

        opts_layout.addStretch()
        layout.addWidget(opts_group)

        # --- Controls ----------------------------------------------------
        btn_layout = QHBoxLayout()
        self._btn_start = QPushButton("Start Inference")
        self._btn_start.clicked.connect(self._start)
        btn_layout.addWidget(self._btn_start)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.clicked.connect(self._stop)
        btn_layout.addWidget(self._btn_stop)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # --- Status ------------------------------------------------------
        self._status = QLabel("Idle")
        layout.addWidget(self._status)

        layout.addStretch()

    # ------------------------------------------------------------------ #
    # Browsing
    # ------------------------------------------------------------------ #
    def _browse_model(self):
        p = self.ask_open_file(
            "Select Model",
            "PyTorch Models (*.pth);;All files (*.*)",
        )
        if p:
            self._edit_model.setText(p)

    def _browse_input(self):
        d = self.ask_directory("Select Input Folder")
        if d:
            self._edit_input.setText(d)

    def _browse_output(self):
        d = self.ask_directory("Select Output Folder")
        if d:
            self._edit_output.setText(d)

    # ------------------------------------------------------------------ #
    # Inference control
    # ------------------------------------------------------------------ #
    def _start(self):
        if self._worker is not None and self._worker.isRunning():
            self.show_error("Inference", "Inference is already running.")
            return

        model_path = self._edit_model.text().strip()
        if not model_path:
            self.show_error("Inference", "Select a model file.")
            return

        input_dir = self._edit_input.text().strip()
        if not input_dir:
            self.show_error("Inference", "Select an input folder.")
            return

        output_dir = self._edit_output.text().strip() or input_dir

        self._status.setText("Loading model...")

        self._worker = InferenceWorker(
            self.inferencer,
            model_path=model_path,
            input_dir=input_dir,
            output_dir=output_dir,
            out_ext=self._combo_ext.currentText(),
            image_size=self._spin_size.value(),
            threshold=self._spin_threshold.value(),
            preserve_depth=self._chk_preserve.isChecked(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.inference_finished.connect(self._on_complete)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _stop(self):
        if self._worker is not None:
            self._worker.stop()
            self._status.setText("Stopping...")

    def _on_worker_finished(self):
        self._worker = None

    # ------------------------------------------------------------------ #
    # Main-thread UI updates
    # ------------------------------------------------------------------ #
    def _on_progress(self, current: int, total: int, fname: str):
        self._status.setText(f"Processed {current}/{total}: {fname}")

    def _on_complete(self, result: dict):
        dur = result.get("duration_seconds", 0.0)
        out_dir = result.get("output_dir", "")
        self.show_info(
            "Inference",
            f"Inference finished in {dur:.1f}s.\nOutput in:\n{out_dir}",
        )
        self._status.setText("Done")

    def _on_error(self, msg: str):
        self.show_error("Inference Error", msg)
        self._status.setText("Idle")

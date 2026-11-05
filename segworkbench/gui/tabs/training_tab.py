"""Tab 1 — UNet Training."""

from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from segworkbench.configs.model_config import ModelConfig
from segworkbench.engine import Trainer
from segworkbench.gui.base_tab import BaseTab
from segworkbench.gui.workers import TrainingWorker
from segworkbench.gui.widgets import MplCanvas


class TrainingTab(BaseTab):
    """Deep-learning training tab with live loss plot and metrics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.trainer = Trainer()

        self._train_losses: list[float] = []
        self._val_losses: list[float] = []
        self._dice_scores: list[float] = []
        self._iou_scores: list[float] = []
        self._worker: TrainingWorker | None = None

        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Dataset -----------------------------------------------------
        data_group = QGroupBox("Dataset")
        data_grid = QGridLayout(data_group)

        data_grid.addWidget(QLabel("Images:"), 0, 0)
        self._edit_image_dir = QLineEdit()
        data_grid.addWidget(self._edit_image_dir, 0, 1)
        btn_browse_img = QPushButton("Browse")
        btn_browse_img.clicked.connect(self._browse_images)
        data_grid.addWidget(btn_browse_img, 0, 2)

        data_grid.addWidget(QLabel("Masks:"), 1, 0)
        self._edit_mask_dir = QLineEdit()
        data_grid.addWidget(self._edit_mask_dir, 1, 1)
        btn_browse_msk = QPushButton("Browse")
        btn_browse_msk.clicked.connect(self._browse_masks)
        data_grid.addWidget(btn_browse_msk, 1, 2)

        data_grid.addWidget(QLabel("Save models to:"), 2, 0)
        self._edit_save_dir = QLineEdit()
        data_grid.addWidget(self._edit_save_dir, 2, 1)
        btn_browse_save = QPushButton("Browse")
        btn_browse_save.clicked.connect(self._browse_save)
        data_grid.addWidget(btn_browse_save, 2, 2)

        data_grid.setColumnStretch(1, 1)
        layout.addWidget(data_group)

        # --- Hyperparameters ---------------------------------------------
        hyp_group = QGroupBox("Hyperparameters")
        hyp_layout = QHBoxLayout(hyp_group)

        hyp_layout.addWidget(QLabel("Epochs:"))
        self._spin_epochs = QSpinBox()
        self._spin_epochs.setRange(1, 10000)
        self._spin_epochs.setValue(5)
        hyp_layout.addWidget(self._spin_epochs)

        hyp_layout.addWidget(QLabel("Batch Size:"))
        self._spin_batch = QSpinBox()
        self._spin_batch.setRange(1, 256)
        self._spin_batch.setValue(2)
        hyp_layout.addWidget(self._spin_batch)

        hyp_layout.addWidget(QLabel("Learning Rate:"))
        self._spin_lr = QDoubleSpinBox()
        self._spin_lr.setDecimals(6)
        self._spin_lr.setRange(1e-8, 1.0)
        self._spin_lr.setSingleStep(1e-4)
        self._spin_lr.setValue(1e-4)
        hyp_layout.addWidget(self._spin_lr)

        hyp_layout.addWidget(QLabel("Workers:"))
        self._spin_workers = QSpinBox()
        self._spin_workers.setRange(0, 16)
        self._spin_workers.setValue(0)
        self._spin_workers.setToolTip(
            "DataLoader workers (0 = load in main thread)"
        )
        hyp_layout.addWidget(self._spin_workers)

        hyp_layout.addStretch()
        layout.addWidget(hyp_group)

        # --- Controls ----------------------------------------------------
        btn_layout = QHBoxLayout()

        self._btn_start = QPushButton("Start Training")
        self._btn_start.clicked.connect(self._start_training)
        btn_layout.addWidget(self._btn_start)

        self._btn_pause = QPushButton("Pause")
        self._btn_pause.clicked.connect(self._pause_training)
        btn_layout.addWidget(self._btn_pause)

        self._btn_resume = QPushButton("Resume")
        self._btn_resume.clicked.connect(self._resume_training)
        btn_layout.addWidget(self._btn_resume)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.clicked.connect(self._stop_training)
        btn_layout.addWidget(self._btn_stop)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Status
        self._status_label = QLabel("Idle")
        layout.addWidget(self._status_label)

        # --- Plots -------------------------------------------------------
        plots_layout = QHBoxLayout()

        self._canvas_loss = MplCanvas(width=4, height=3, dpi=100)
        self._canvas_loss.axes.set_xlabel("Epoch")
        self._canvas_loss.axes.set_ylabel("Loss")
        self._line_loss, = self._canvas_loss.axes.plot([], [], label="Train Loss")
        self._line_val, = self._canvas_loss.axes.plot([], [], label="Val Loss")
        self._canvas_loss.axes.legend(loc="upper right")
        plots_layout.addWidget(self._canvas_loss)

        self._canvas_metrics = MplCanvas(width=4, height=3, dpi=100)
        self._canvas_metrics.axes.set_xlabel("Epoch")
        self._canvas_metrics.axes.set_ylabel("Score")
        self._line_dice, = self._canvas_metrics.axes.plot([], [], label="Dice")
        self._line_iou, = self._canvas_metrics.axes.plot([], [], label="IoU")
        self._canvas_metrics.axes.legend(loc="lower right")
        plots_layout.addWidget(self._canvas_metrics)

        layout.addLayout(plots_layout, stretch=1)

    # ------------------------------------------------------------------ #
    # Browsing
    # ------------------------------------------------------------------ #
    def _browse_images(self):
        d = self.ask_directory("Select Training Images Folder")
        if d:
            self._edit_image_dir.setText(d)

    def _browse_masks(self):
        d = self.ask_directory("Select Training Masks Folder")
        if d:
            self._edit_mask_dir.setText(d)

    def _browse_save(self):
        d = self.ask_directory("Select Folder to Save Models")
        if d:
            self._edit_save_dir.setText(d)

    # ------------------------------------------------------------------ #
    # Training control
    # ------------------------------------------------------------------ #
    def _start_training(self):
        if self._worker is not None and self._worker.isRunning():
            self.show_error("Training", "Training is already running.")
            return

        image_dir = self._edit_image_dir.text().strip()
        mask_dir = self._edit_mask_dir.text().strip()
        save_dir = self._edit_save_dir.text().strip()

        if not all((image_dir, mask_dir, save_dir)):
            self.show_error(
                "Training",
                "Please select image folder, mask folder, and save folder.",
            )
            return

        # Reset state
        self._train_losses.clear()
        self._val_losses.clear()
        self._dice_scores.clear()
        self._iou_scores.clear()
        self._line_loss.set_data([], [])
        self._line_val.set_data([], [])
        self._line_dice.set_data([], [])
        self._line_iou.set_data([], [])
        self._canvas_loss.axes.relim()
        self._canvas_loss.axes.autoscale_view()
        self._canvas_loss.draw_idle()
        self._canvas_metrics.axes.relim()
        self._canvas_metrics.axes.autoscale_view()
        self._canvas_metrics.draw_idle()
        self._status_label.setText("Training...")

        config = ModelConfig(
            epochs=self._spin_epochs.value(),
            batch_size=self._spin_batch.value(),
            learning_rate=self._spin_lr.value(),
            image_size=(256, 256),
        )

        self._worker = TrainingWorker(
            self.trainer,
            image_dir=image_dir,
            mask_dir=mask_dir,
            save_dir=save_dir,
            config=config,
            num_workers=self._spin_workers.value(),
        )
        self._worker.epoch_finished.connect(self._update_plot)
        self._worker.training_finished.connect(self._handle_result)
        self._worker.error.connect(self._handle_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _pause_training(self):
        if self._worker is not None:
            self._worker.pause()
            self._status_label.setText("Paused.")

    def _resume_training(self):
        if self._worker is not None:
            self._worker.resume()
            self._status_label.setText("Resumed.")

    def _stop_training(self):
        if self._worker is not None:
            self._worker.stop()
            self._status_label.setText("Stopping...")

    def _on_worker_finished(self):
        self._worker = None

    # ------------------------------------------------------------------ #
    # Main-thread UI updates
    # ------------------------------------------------------------------ #
    def _update_plot(self, data: dict):
        epoch = data["epoch"]
        total = data["total"]
        tr = data["train_loss"]
        vl = data["val_loss"]
        dice = data["dice"]
        iou = data["iou"]

        self._train_losses.append(tr)
        self._val_losses.append(vl)
        self._dice_scores.append(dice)
        self._iou_scores.append(iou)

        xs = list(range(1, len(self._train_losses) + 1))

        self._line_loss.set_data(xs, self._train_losses)
        self._line_val.set_data(xs, self._val_losses)
        self._canvas_loss.axes.relim()
        self._canvas_loss.axes.autoscale_view()
        self._canvas_loss.draw_idle()

        self._line_dice.set_data(xs, self._dice_scores)
        self._line_iou.set_data(xs, self._iou_scores)
        self._canvas_metrics.axes.relim()
        self._canvas_metrics.axes.autoscale_view()
        self._canvas_metrics.draw_idle()

        self._status_label.setText(
            f"Epoch {epoch}/{total} | Loss: {tr:.4f}/{vl:.4f} | "
            f"Dice: {dice:.4f} | IoU: {iou:.4f}"
        )

    def _handle_result(self, result: dict, was_stopped: bool):
        if was_stopped:
            self.show_info("Training", "Training stopped by user.")
        else:
            msg = (
                f"Training complete.\n"
                f"Best val loss: {result['best_val_loss']:.4f}\n"
                f"Best Dice: {result.get('best_dice', 0.0):.4f}\n"
                f"Best IoU: {result.get('best_iou', 0.0):.4f}\n"
                f"Model saved:\n{result['best_model_path']}"
            )
            self.show_info("Training", msg)
        self._status_label.setText("Idle")

    def _handle_error(self, msg: str):
        self.show_error("Training Error", msg)
        self._status_label.setText("Idle")

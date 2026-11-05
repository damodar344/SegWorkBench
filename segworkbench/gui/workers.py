"""Qt-friendly worker threads for long-running engine tasks."""

import logging
import threading

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class TrainingWorker(QThread):
    """Run ``Trainer.train()`` in a background QThread."""

    # Pass epoch data as a plain dict to avoid C++ type-conversion edge cases.
    epoch_finished = Signal(object)  # dict with keys: epoch, total, train_loss, val_loss, dice, iou
    training_finished = Signal(object, bool)  # result dict, was_stopped
    error = Signal(str)

    def __init__(self, trainer, **kwargs):
        super().__init__()
        self.trainer = trainer
        self.kwargs = kwargs
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

    def run(self):
        logger.info("TrainingWorker started")
        try:
            result = self.trainer.train(
                stop_event=self._stop_event,
                pause_event=self._pause_event,
                on_epoch_end=lambda e, t, tr, vl, dc, iu: self.epoch_finished.emit(
                    {
                        "epoch": e,
                        "total": t,
                        "train_loss": tr,
                        "val_loss": vl,
                        "dice": dc,
                        "iou": iu,
                    }
                ),
                **self.kwargs,
            )
            was_stopped = self._stop_event.is_set()
            logger.info("TrainingWorker finished (stopped=%s)", was_stopped)
            self.training_finished.emit(result, was_stopped)
        except Exception as exc:
            logger.exception("Training worker failed")
            self.error.emit(str(exc))

    def stop(self):
        self._stop_event.set()

    def pause(self):
        self._pause_event.set()

    def resume(self):
        self._pause_event.clear()


class InferenceWorker(QThread):
    """Run ``Inferencer.run()`` in a background QThread."""

    progress = Signal(int, int, str)  # current, total, filename
    inference_finished = Signal(dict)  # result
    error = Signal(str)

    def __init__(self, inferencer, **kwargs):
        super().__init__()
        self.inferencer = inferencer
        self.kwargs = kwargs
        self._stop = False

    def run(self):
        logger.info("InferenceWorker started")
        try:
            result = self.inferencer.run(
                should_stop=lambda: self._stop,
                on_progress=lambda i, t, f: self.progress.emit(i, t, f),
                **self.kwargs,
            )
            logger.info("InferenceWorker finished")
            self.inference_finished.emit(result)
        except Exception as exc:
            logger.exception("Inference worker failed")
            self.error.emit(str(exc))

    def stop(self):
        self._stop = True

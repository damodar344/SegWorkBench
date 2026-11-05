"""Base tab widget with shared helpers (dialogs, safe cross-thread calls)."""

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtWidgets import (
    QWidget,
    QMessageBox,
    QFileDialog,
)


class _SafeCallHelper(QObject):
    """Internal helper to marshal callable invocations onto the Qt event loop."""

    invoke = Signal(object, tuple, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.invoke.connect(self._do_call, Qt.QueuedConnection)

    def _do_call(self, fn, args, kwargs):
        fn(*args, **kwargs)


class BaseTab(QWidget):
    """Base class for all application tabs.

    Provides:
    - Convenience wrappers for file/folder dialogs.
    - Info / error message boxes.
    - ``safe_call(fn, *args, **kwargs)`` to run a callback on the main (GUI) thread
      from a worker thread.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._safe_helper = _SafeCallHelper(self)

    # ------------------------------------------------------------------ #
    # Dialog helpers
    # ------------------------------------------------------------------ #
    def show_info(self, title: str, msg: str) -> None:
        QMessageBox.information(self, title, msg)

    def show_error(self, title: str, msg: str) -> None:
        QMessageBox.critical(self, title, msg)

    def ask_directory(self, title: str = "Select Folder") -> str:
        """Return the selected directory path or an empty string."""
        d = QFileDialog.getExistingDirectory(self, title)
        return d or ""

    def ask_open_file(self, title: str = "Select File", filters: str = "All files (*.*)") -> str:
        """Return the selected file path or an empty string."""
        p, _ = QFileDialog.getOpenFileName(self, title, "", filters)
        return p or ""

    def ask_save_file(
        self,
        title: str = "Save File",
        filters: str = "All files (*.*)",
        default_name: str = "",
    ) -> str:
        """Return the chosen save path or an empty string."""
        p, _ = QFileDialog.getSaveFileName(self, title, default_name, filters)
        return p or ""

    # ------------------------------------------------------------------ #
    # Thread-safety
    # ------------------------------------------------------------------ #
    def safe_call(self, fn, *args, **kwargs) -> None:
        """Schedule *fn* to run on the main GUI thread.

        This is safe to call from any Python ``threading.Thread``.
        """
        self._safe_helper.invoke.emit(fn, args, kwargs)

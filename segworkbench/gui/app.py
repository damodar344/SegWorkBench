"""Main application window."""

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMenu,
    QStatusBar,
    QTabWidget,
)

from segworkbench.configs import AppConfig
from segworkbench.gui.tabs import TrainingTab, InferenceTab, GMMTab, MaskEditorTab


def _load_stylesheet(name: str) -> str:
    """Load a bundled stylesheet by *name* (dark | light)."""
    qss_path = os.path.join(
        os.path.dirname(__file__), "styles", f"{name}.qss"
    )
    if os.path.isfile(qss_path):
        with open(qss_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


class MainWindow(QMainWindow):
    """Primary application window with a tabbed interface."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Segmentation Suite — PySide6")
        self.resize(1200, 750)

        self._app_config = AppConfig().load_default()

        # --- Menu bar ---
        self._build_menu()

        # --- Tabs ---
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tabs.addTab(TrainingTab(), "1) Training (UNet)")
        self.tabs.addTab(InferenceTab(), "2) Inference")
        self.tabs.addTab(GMMTab(), "3) GMM")
        self.tabs.addTab(MaskEditorTab(), "4) Mask Editor")

        # --- Status bar ---
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

    # ------------------------------------------------------------------ #
    # Theme
    # ------------------------------------------------------------------ #
    def set_theme(self, app: QApplication, name: str) -> None:
        """Apply the named stylesheet to *app* and persist the choice."""
        sheet = _load_stylesheet(name)
        app.setStyleSheet(sheet)
        self._app_config.theme = name
        self._app_config.save_default()

    @classmethod
    def apply_theme(cls, app: QApplication) -> None:
        """Apply the currently saved (or default dark) theme."""
        cfg = AppConfig().load_default()
        sheet = _load_stylesheet(cfg.theme)
        app.setStyleSheet(sheet)

    # ------------------------------------------------------------------ #
    # Menu
    # ------------------------------------------------------------------ #
    def _build_menu(self):
        menubar = self.menuBar()

        # File
        file_menu = menubar.addMenu("&File")
        exit_action = file_menu.addAction("E&xit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)

        # View — Theme
        view_menu = menubar.addMenu("&View")
        theme_menu = view_menu.addMenu("&Theme")
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)

        for label, theme_name in [("&Dark", "dark"), ("&Light", "light")]:
            act = theme_menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(self._app_config.theme == theme_name)
            act.triggered.connect(
                lambda _checked, tn=theme_name: self._on_theme_changed(tn)
            )
            theme_group.addAction(act)

        # Help
        help_menu = menubar.addMenu("&Help")
        about_action = help_menu.addAction("&About")
        about_action.triggered.connect(self._show_about)

    def _on_theme_changed(self, theme_name: str):
        app = QApplication.instance()
        if app is not None:
            self.set_theme(app, theme_name)

    def _show_about(self):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(
            self,
            "About SegWorkbench",
            "<b>SegWorkbench</b><br>"
            "Image Segmentation Workbench<br><br>"
            "Built with PySide6, PyTorch, and scikit-learn.<br>"
            "© 2024 SegWorkbench authors.",
        )

    # ------------------------------------------------------------------ #
    # Keyboard shortcuts
    # ------------------------------------------------------------------ #
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            self.toggle_fullscreen()
        elif event.key() == Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
        else:
            super().keyPressEvent(event)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

"""Entry point: python -m segworkbench"""

import sys

from PySide6.QtWidgets import QApplication

from segworkbench.gui.app import MainWindow


def main():
    app = QApplication(sys.argv)
    MainWindow.apply_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

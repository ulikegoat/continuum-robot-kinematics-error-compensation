"""Stable entry point for the redesigned continuum-robot research dashboard.

Run manually with ``python -m continuum_robot.gui.pcc_gui``.
"""
from __future__ import annotations

import sys

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

from continuum_robot.gui.main_window import MainWindow
from continuum_robot.gui.theme import apply_theme


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Continuum Lab")
    app.setOrganizationName("Continuum Robot Kinematics")
    apply_theme(app)
    pg.setConfigOptions(antialias=True)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

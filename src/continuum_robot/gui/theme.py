"""Warm, flat visual system for the Continuum Lab engineering workstation."""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
import pyqtgraph as pg


COLORS = {
    "bg": "#E9E8E4", "surface": "#F6F5F2", "surface2": "#DEDDD8",
    "border": "#C7C5BF", "text": "#242424", "muted": "#70706C",
    "accent": "#B86E42", "positive": "#397C61", "danger": "#B84F49",
    "viewport": "#17191B", "pcc": "#B86E42", "reference": "#906A36",
    "ffnn": "#397C61", "knn": "#765E9B", "linear": "#62676A",
    "ridge2": "#A65061", "ridge3": "#9C782E",
}


def apply_theme(app) -> None:
    pg.setConfigOptions(background=COLORS["surface"], foreground=COLORS["text"], antialias=True)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 9))
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLORS["bg"]))
    palette.setColor(QPalette.WindowText, QColor(COLORS["text"]))
    palette.setColor(QPalette.Base, QColor(COLORS["surface"]))
    palette.setColor(QPalette.AlternateBase, QColor(COLORS["surface2"]))
    palette.setColor(QPalette.Text, QColor(COLORS["text"]))
    palette.setColor(QPalette.Button, QColor(COLORS["surface2"]))
    palette.setColor(QPalette.ButtonText, QColor(COLORS["text"]))
    palette.setColor(QPalette.Highlight, QColor(COLORS["accent"]))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)
    app.setStyleSheet(f"""
        QWidget {{ color: {COLORS['text']}; background: transparent; }}
        QMainWindow, QWidget#Root {{ background: {COLORS['bg']}; }}
        QFrame#Toolbar {{ background: {COLORS['surface']}; border-bottom: 1px solid {COLORS['border']}; }}
        QFrame#Navigation {{ background: {COLORS['surface2']}; border-right: 1px solid {COLORS['border']}; }}
        QFrame#Inspector, QFrame#Panel, QFrame#Telemetry {{ background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; }}
        QFrame#Inspector {{ border-top: 0; border-bottom: 0; border-right: 0; }}
        QFrame#Telemetry {{ border-left: 0; border-right: 0; border-bottom: 0; }}
        QFrame#TelemetryCell {{ border-right: 1px solid {COLORS['border']}; }}
        QLabel#Title {{ font-size: 15px; font-weight: 700; }}
        QLabel#SectionTitle {{ font-size: 11px; font-weight: 700; }}
        QLabel#Eyebrow, QLabel#CardLabel {{ color: {COLORS['muted']}; font-size: 9px; font-weight: 600; }}
        QLabel#Muted {{ color: {COLORS['muted']}; }}
        QLabel#CardValue {{ font-family: 'Cascadia Code', Consolas, monospace; font-size: 11px; font-weight: 600; }}
        QLabel#Status {{ color: {COLORS['positive']}; font-weight: 600; }}
        QComboBox, QDoubleSpinBox, QSpinBox {{ background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; border-radius: 2px; padding: 3px 6px; min-height: 20px; }}
        QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {{ border-color: {COLORS['accent']}; }}
        QComboBox::drop-down {{ border: 0; width: 20px; }}
        QComboBox QAbstractItemView {{ background: {COLORS['surface']}; color: {COLORS['text']}; selection-background-color: {COLORS['surface2']}; selection-color: {COLORS['text']}; border: 1px solid {COLORS['border']}; }}
        QPushButton {{ background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; border-radius: 2px; padding: 5px 9px; font-weight: 600; }}
        QPushButton:hover {{ background: {COLORS['surface2']}; border-color: {COLORS['accent']}; }}
        QPushButton:pressed {{ background: {COLORS['border']}; }}
        QPushButton:disabled, QComboBox:disabled, QSpinBox:disabled {{ color: {COLORS['muted']}; background: {COLORS['surface2']}; }}
        QPushButton#Primary {{ color: #FFFFFF; background: {COLORS['accent']}; border-color: {COLORS['accent']}; }}
        QPushButton#Primary:hover {{ background: #9B5935; }}
        QPushButton#Nav {{ background: transparent; border: 0; border-left: 3px solid transparent; border-radius: 0; text-align: left; padding: 11px 6px; color: {COLORS['muted']}; }}
        QPushButton#Nav:hover {{ background: {COLORS['bg']}; color: {COLORS['text']}; }}
        QPushButton#Nav:checked {{ background: {COLORS['surface']}; color: {COLORS['text']}; border-left: 3px solid {COLORS['accent']}; }}
        QPushButton#Group {{ background: transparent; border: 0; border-bottom: 1px solid {COLORS['border']}; border-radius: 0; text-align: left; padding: 8px 2px; }}
        QPushButton#ViewportTool {{ background: #282A2B; color: #E9E8E4; border: 1px solid #555653; padding: 4px 7px; }}
        QPushButton#ViewportTool:hover {{ background: #383A39; border-color: #B86E42; }}
        QTabWidget::pane {{ border: 1px solid {COLORS['border']}; background: {COLORS['surface']}; }}
        QTabBar::tab {{ background: {COLORS['surface2']}; color: {COLORS['muted']}; padding: 7px 11px; margin-right: 1px; border: 1px solid {COLORS['border']}; border-bottom: 0; }}
        QTabBar::tab:selected {{ color: {COLORS['text']}; background: {COLORS['surface']}; border-top: 2px solid {COLORS['accent']}; }}
        QCheckBox {{ spacing: 6px; }}
        QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {COLORS['border']}; border-radius: 1px; background: {COLORS['surface']}; }}
        QCheckBox::indicator:checked {{ background: {COLORS['accent']}; border-color: {COLORS['accent']}; }}
        QSlider::groove:horizontal {{ background: {COLORS['border']}; height: 4px; }}
        QSlider::handle:horizontal {{ background: {COLORS['accent']}; width: 11px; margin: -5px 0; border: 1px solid #8F5434; border-radius: 2px; }}
        QScrollArea {{ border: 0; background: {COLORS['surface']}; }}
        QScrollBar:vertical {{ background: {COLORS['surface']}; width: 8px; }}
        QScrollBar::handle:vertical {{ background: {COLORS['border']}; min-height: 26px; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        QSplitter::handle {{ background: {COLORS['border']}; }}
        QToolTip {{ background: {COLORS['surface']}; color: {COLORS['text']}; border: 1px solid {COLORS['border']}; }}
        QStatusBar {{ background: {COLORS['surface']}; border-top: 1px solid {COLORS['border']}; }}
    """)

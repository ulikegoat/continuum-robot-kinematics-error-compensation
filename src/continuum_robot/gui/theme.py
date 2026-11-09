"""Visual language for the thesis dashboard."""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette

COLORS = {
    "bg": "#080F1B", "surface": "#101C2B", "surface2": "#152539",
    "border": "#263A50", "text": "#EBF3FA", "muted": "#91A7BA",
    "pcc": "#F3A46B", "reference": "#9FB5CE", "ffnn": "#4FD1B5",
    "knn": "#9F8CFF", "accent": "#55B8E7", "danger": "#F18186",
}


def apply_theme(app) -> None:
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLORS["bg"]))
    palette.setColor(QPalette.WindowText, QColor(COLORS["text"]))
    palette.setColor(QPalette.Base, QColor(COLORS["surface"]))
    palette.setColor(QPalette.AlternateBase, QColor(COLORS["surface2"]))
    palette.setColor(QPalette.Text, QColor(COLORS["text"]))
    palette.setColor(QPalette.Button, QColor(COLORS["surface2"]))
    palette.setColor(QPalette.ButtonText, QColor(COLORS["text"]))
    palette.setColor(QPalette.Highlight, QColor(COLORS["accent"]))
    app.setPalette(palette)
    app.setStyleSheet(f"""
        QWidget {{ color: {COLORS['text']}; background: transparent; }}
        QMainWindow, QWidget#Root {{ background: {COLORS['bg']}; }}
        QFrame#Panel, QFrame#MetricCard {{ background: {COLORS['surface']};
            border: 1px solid {COLORS['border']}; border-radius: 14px; }}
        QFrame#Header {{ background: {COLORS['surface']}; border-bottom: 1px solid {COLORS['border']}; }}
        QLabel#Eyebrow {{ color: {COLORS['accent']}; font-size: 10px; font-weight: 700; letter-spacing: 2px; }}
        QLabel#Title {{ font-size: 21px; font-weight: 700; }}
        QLabel#SectionTitle {{ font-size: 13px; font-weight: 700; }}
        QLabel#Muted, QLabel#CardLabel {{ color: {COLORS['muted']}; }}
        QLabel#CardValue {{ font-family: 'Cascadia Code', Consolas, monospace; font-size: 16px; font-weight: 700; }}
        QLabel#Status {{ color: {COLORS['accent']}; font-weight: 600; }}
        QComboBox, QDoubleSpinBox, QSpinBox {{ background: {COLORS['surface2']};
            border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 7px 10px; min-height: 22px; }}
        QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {{ border-color: {COLORS['accent']}; }}
        QComboBox::drop-down {{ border: 0; width: 22px; }}
        QPushButton {{ background: {COLORS['surface2']}; border: 1px solid {COLORS['border']};
            border-radius: 8px; padding: 8px 12px; font-weight: 600; }}
        QPushButton:hover {{ background: #1C344A; border-color: {COLORS['accent']}; }}
        QPushButton:pressed {{ background: #294762; }}
        QPushButton:checked, QPushButton#Primary {{ background: #164B61; border-color: {COLORS['accent']}; }}
        QPushButton:disabled {{ color: #667C8F; }}
        QTabWidget::pane {{ border: 1px solid {COLORS['border']}; border-radius: 9px; }}
        QTabBar::tab {{ background: {COLORS['surface']}; color: {COLORS['muted']};
            padding: 8px 15px; margin-right: 3px; border-radius: 7px; }}
        QTabBar::tab:selected {{ color: {COLORS['text']}; background: {COLORS['surface2']}; }}
        QCheckBox {{ spacing: 8px; color: {COLORS['muted']}; }}
        QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {COLORS['border']}; border-radius: 4px; background: {COLORS['surface2']}; }}
        QCheckBox::indicator:checked {{ background: {COLORS['accent']}; border-color: {COLORS['accent']}; }}
        QSlider::groove:horizontal {{ background: #29435B; height: 5px; border-radius: 2px; }}
        QSlider::handle:horizontal {{ background: {COLORS['accent']}; width: 16px; margin: -6px 0; border-radius: 8px; }}
        QScrollArea {{ border: 0; }}
        QSplitter::handle {{ background: {COLORS['border']}; }}
        QToolTip {{ background: {COLORS['surface2']}; color: {COLORS['text']}; border: 1px solid {COLORS['accent']}; }}
    """)

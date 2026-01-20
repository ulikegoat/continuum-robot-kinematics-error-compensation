import sys
from pathlib import Path
import numpy as np
import joblib

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QComboBox, QPushButton, QFrame, QCheckBox, QMessageBox,
    QButtonGroup
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase, QColor
from PySide6.QtWidgets import QGraphicsDropShadowEffect

import pyqtgraph as pg
import pyqtgraph.opengl as gl

import pcc_model as pcc
import real_model as real


MODELS_DIR = Path("models")  # expected: Linear.pkl, PolyDeg3.pkl, KNN.pkl

# Centralized theme palette for consistent UI styling
COL = {
    "BG": "#070A14",
    "PANEL_BG": (18, 26, 52, 195),   # rgba
    "PANEL_BG2": (10, 16, 34, 195),
    "BORDER": "rgba(140, 160, 255, 55)",
    "BORDER_HI": "rgba(140, 160, 255, 110)",

    "TEXT": "rgba(235, 241, 255, 235)",
    "MUTED": "rgba(170, 185, 220, 190)",

    "PCC": "#FF9A7A",       # PCC curve color
    "REAL": "#A8B4D8",      # REAL curve color
    "ML": "#9B7CFF",        # ML tip color

    "POS": "#4ADE80",       # positive sign color
    "NEG": "#FB7185",       # negative sign color
    "WARN": "#FBBF24",
}


def tip_xyz(X, Y, Z):
    # Tip position is the last point of the generated centerline
    return np.array([float(X[-1]), float(Y[-1]), float(Z[-1])], dtype=float)


def safe_real_shape(l1, l2, l3):
    # Call real.real_shape with enforce_limit if the model supports it
    try:
        return real.real_shape(l1, l2, l3, enforce_limit=True)
    except TypeError:
        return real.real_shape(l1, l2, l3)


def add_shadow(widget: QWidget, blur=28, y_off=8, alpha=90, color=(90, 110, 255)):
    # Drop shadow used to fake "glass / product UI" depth
    eff = QGraphicsDropShadowEffect()
    eff.setBlurRadius(blur)
    eff.setOffset(0, y_off)
    eff.setColor(QColor(color[0], color[1], color[2], alpha))
    widget.setGraphicsEffect(eff)


class Card(QFrame):
    # Small metric card: title + big value
    def __init__(self, title: str, value: str = ""):
        super().__init__()
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(2)

        self.t = QLabel(title)
        self.t.setObjectName("CardTitle")

        self.v = QLabel(value)
        self.v.setObjectName("MetricValue")

        lay.addWidget(self.t)
        lay.addWidget(self.v)

        add_shadow(self, blur=32, y_off=10, alpha=70)

    def set_value(self, s: str):
        self.v.setText(s)


class GlassPanel(QFrame):
    # Semi-transparent container panel used throughout the UI
    def __init__(self, title: str | None = None):
        super().__init__()
        self.setObjectName("Panel")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(14, 14, 14, 14)
        self.lay.setSpacing(10)

        if title is not None:
            lbl = QLabel(title)
            lbl.setObjectName("PanelTitle")
            self.lay.addWidget(lbl)

        add_shadow(self, blur=42, y_off=14, alpha=55)


class DlBar(QFrame):
    # Visual indicator for Δl: label + fill bar + signed value text
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("DlRow")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        self.lbl = QLabel(title)
        self.lbl.setObjectName("DlTitle")
        self.lbl.setMinimumWidth(72)

        self.bar = QFrame()
        self.bar.setObjectName("DlBar")

        self.fill = QFrame(self.bar)
        self.fill.setObjectName("DlFill")
        self.fill.setGeometry(0, 0, 0, 8)

        self.val = QLabel("+0.00")
        self.val.setObjectName("DlValue")
        self.val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.val.setMinimumWidth(70)

        lay.addWidget(self.lbl)
        lay.addWidget(self.bar, stretch=1)
        lay.addWidget(self.val)

        self._max = 10.0  # bar normalization scale in mm

    def set_value(self, x: float):
        # Display signed value and color-code by sign
        sign = "+" if x >= 0 else "−"
        self.val.setText(f"{sign}{abs(x):.2f} mm")

        c = COL["POS"] if x >= 0 else COL["NEG"]
        self.val.setStyleSheet(f"color: {c};")

        # Fill length proportional to |x| with saturation at max
        r = min(abs(x) / self._max, 1.0)
        w = max(int(self.bar.width() * r), 0)
        self.fill.setGeometry(0, 0, w, 8)
        self.fill.setStyleSheet(f"background: {c}; border-radius: 4px;")

    def resizeEvent(self, ev):
        # Keep the fill bar height consistent when resizing
        self.fill.setGeometry(self.fill.geometry().x(), 0, self.fill.width(), 8)
        super().resizeEvent(ev)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Continuum Robot GUI — PCC / REAL / ML Compensation")
        self.resize(1400, 800)

        self.current_proj = "XZ"                 # active 2D projection mode
        self.ml_model = None                     # loaded sklearn pipeline (or None)
        self.ml_bias = np.zeros(3, dtype=float)  # shift so ML correction is 0 at dl=(0,0,0)

        # Root container
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(16, 16, 16, 16)
        main.setSpacing(12)

        # Top metric cards
        top = QHBoxLayout()
        top.setSpacing(12)
        self.card_theta = Card("θ", "0.0°")
        self.card_e_pcc = Card("|REAL − PCC|", "0.000 mm")
        self.card_e_ml  = Card("|REAL − (PCC+ML)|", "0.000 mm")
        self.card_limit = Card("Constraint", "Max 2 tendons")
        top.addWidget(self.card_theta)
        top.addWidget(self.card_e_pcc)
        top.addWidget(self.card_e_ml)
        top.addWidget(self.card_limit)
        main.addLayout(top)

        # Main area: 3D view + 2D projection + control panel
        mid = QHBoxLayout()
        mid.setSpacing(12)
        main.addLayout(mid, stretch=1)

        # 3D panel (pyqtgraph.opengl)
        left_panel = GlassPanel("3D View")
        mid.addWidget(left_panel, stretch=3)

        self.gl = gl.GLViewWidget()
        self.gl.setBackgroundColor((10, 14, 26, 255))
        self.gl.opts["distance"] = 220
        self.gl.opts["elevation"] = 20
        self.gl.opts["azimuth"] = -45
        left_panel.lay.addWidget(self.gl, stretch=1)

        # Ground grid for spatial context
        self.grid = gl.GLGridItem()
        self.grid.setSize(180, 180)
        self.grid.setSpacing(25, 25)
        self.grid.translate(0, 0, 0)
        self.gl.addItem(self.grid)

        # 3D centerlines and ML tip marker
        self.line3d_pcc = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)),
            mode="line_strip",
            antialias=True,
            color=(1.0, 0.60, 0.48, 0.90),
            width=3
        )
        self.line3d_real = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)),
            mode="line_strip",
            antialias=True,
            color=(0.66, 0.71, 0.85, 0.32),
            width=3
        )
        self.tip3d_ml = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            size=11,
            color=(0.61, 0.49, 1.00, 1.0)
        )
        self.gl.addItem(self.line3d_pcc)
        self.gl.addItem(self.line3d_real)
        self.gl.addItem(self.tip3d_ml)

        # XYZ axes are always visible (debug + orientation)
        axis_len = 120.0
        self.x_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [axis_len, 0, 0]], dtype=float),
            color=(1.0, 0.35, 0.35, 0.85), width=2, antialias=True
        )
        self.y_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, axis_len, 0]], dtype=float),
            color=(0.35, 1.0, 0.45, 0.85), width=2, antialias=True
        )
        self.z_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, axis_len]], dtype=float),
            color=(0.40, 0.70, 1.0, 0.85), width=2, antialias=True
        )
        self.gl.addItem(self.x_axis)
        self.gl.addItem(self.y_axis)
        self.gl.addItem(self.z_axis)

        # 2D panel (pyqtgraph)
        center_panel = GlassPanel("2D Projection")
        mid.addWidget(center_panel, stretch=3)

        self.plot2d = pg.PlotWidget()
        self.plot2d.setBackground((10, 14, 26))
        self.plot2d.showGrid(x=True, y=True, alpha=0.10)

        axL = self.plot2d.getAxis("left")
        axB = self.plot2d.getAxis("bottom")
        axL.setTextPen(pg.mkPen("#D7E7FF"))
        axB.setTextPen(pg.mkPen("#D7E7FF"))
        axL.setPen(pg.mkPen((160, 180, 255, 70)))
        axB.setPen(pg.mkPen((160, 180, 255, 70)))

        # Disable context menu and clip curves to view for performance
        self.plot2d.getPlotItem().getViewBox().setMenuEnabled(False)
        self.plot2d.getPlotItem().setClipToView(True)

        center_panel.lay.addWidget(self.plot2d, stretch=1)

        # 2D curves + ML tip marker
        self.curve2d_pcc = self.plot2d.plot([], [], pen=pg.mkPen(COL["PCC"], width=2.2))
        self.curve2d_real = self.plot2d.plot([], [], pen=pg.mkPen((168, 180, 216, 85), width=2.0))
        self.tip2d_ml = self.plot2d.plot(
            [], [],
            pen=None,
            symbol="o",
            symbolSize=11,
            symbolBrush=pg.mkBrush(COL["ML"]),
            symbolPen=pg.mkPen((255, 255, 255, 40))
        )

        # Controls panel
        right_panel = GlassPanel("Controls")
        mid.addWidget(right_panel, stretch=2)

        # Model selector (loads a sklearn pipeline from ./models)
        lab_ml = QLabel("ML model (auto-load)")
        lab_ml.setObjectName("SectionLabel")
        right_panel.lay.addWidget(lab_ml)

        self.model_box = QComboBox()
        self.model_box.currentTextChanged.connect(self.on_model_changed)
        right_panel.lay.addWidget(self.model_box)

        # Projection selector buttons
        lab_proj = QLabel("Projection")
        lab_proj.setObjectName("SectionLabel")
        right_panel.lay.addWidget(lab_proj)

        proj_row = QHBoxLayout()
        proj_row.setSpacing(8)
        self.proj_group = QButtonGroup(self)
        self.proj_group.setExclusive(True)

        self.btn_proj = {}
        for lab in ["XZ", "XY", "YZ"]:
            b = QPushButton(lab)
            b.setCheckable(True)
            b.setObjectName("SegBtn")
            if lab == "XZ":
                b.setChecked(True)
            b.clicked.connect(lambda _, x=lab: self.set_projection(x))
            self.proj_group.addButton(b)
            self.btn_proj[lab] = b
            proj_row.addWidget(b)

        right_panel.lay.addLayout(proj_row)

        # Visibility toggles (PCC / REAL / PCC+ML)
        lab_leg = QLabel("Legend / Visibility")
        lab_leg.setObjectName("SectionLabel")
        right_panel.lay.addWidget(lab_leg)

        self.cb_show_pcc = QCheckBox("Show PCC")
        self.cb_show_real = QCheckBox("Show REAL")
        self.cb_show_ml = QCheckBox("Show PCC+ML")
        self.cb_show_pcc.setChecked(True)
        self.cb_show_real.setChecked(True)
        self.cb_show_ml.setChecked(True)
        for cb in (self.cb_show_pcc, self.cb_show_real, self.cb_show_ml):
            cb.stateChanged.connect(self.update_scene)
            right_panel.lay.addWidget(cb)

        # Tendon sliders + Δl indicators
        lab_dl = QLabel("Tendons (Δl)")
        lab_dl.setObjectName("SectionLabel")
        right_panel.lay.addWidget(lab_dl)

        self.s1 = self.make_slider()
        self.s2 = self.make_slider()
        self.s3 = self.make_slider()

        self.dl1 = DlBar("Δl1")
        self.dl2 = DlBar("Δl2")
        self.dl3 = DlBar("Δl3")

        # Group each bar with its slider for readability
        def add_dl_block(slider: QSlider, bar: DlBar):
            wrap = QVBoxLayout()
            wrap.setSpacing(6)
            wrap.addWidget(bar)
            wrap.addWidget(slider)
            right_panel.lay.addLayout(wrap)

        add_dl_block(self.s1, self.dl1)
        add_dl_block(self.s2, self.dl2)
        add_dl_block(self.s3, self.dl3)

        right_panel.lay.addStretch(1)

        # Bottom hint for usage and what ML is compensating
        hint = QLabel("Tip: use 2 tendons max (constraint is enforced). ML compensates TIP error (dX,dY,dZ).")
        hint.setObjectName("Hint")
        main.addWidget(hint)

        # Load models list and initialize first selection
        self.refresh_models()
        self.on_model_changed(self.model_box.currentText())
        self.update_scene()

    # UI helpers
    def make_slider(self):
        # Slider uses integer ticks 0..1000 mapped to 0..10 mm
        s = QSlider(Qt.Horizontal)
        s.setMinimum(0)
        s.setMaximum(1000)
        s.setValue(0)
        s.valueChanged.connect(self.update_scene)
        s.setObjectName("DlSlider")
        return s

    def dl_values(self):
        # Convert slider positions to tendon shortenings in mm
        return (self.s1.value() / 100.0, self.s2.value() / 100.0, self.s3.value() / 100.0)

    def enforce_max2(self, l1, l2, l3):
        # If more than 2 tendons are active, zero the smallest non-zero value
        dls = np.array([l1, l2, l3], dtype=float)
        active = np.sum(dls > 1e-6)
        if active <= 2:
            return l1, l2, l3, False

        idx = np.argsort(dls)
        for i in idx:
            if dls[i] > 1e-6:
                dls[i] = 0.0
                break
        return float(dls[0]), float(dls[1]), float(dls[2]), True

    def set_projection(self, proj: str):
        # Switch which 2D plane is shown
        self.current_proj = proj
        if proj in self.btn_proj:
            self.btn_proj[proj].setChecked(True)
        self.update_scene()

    # Model loading
    def refresh_models(self):
        # Populate dropdown from *.pkl files in ./models with preferred ordering
        self.model_box.blockSignals(True)
        self.model_box.clear()

        candidates = []
        if MODELS_DIR.exists():
            for p in MODELS_DIR.glob("*.pkl"):
                candidates.append(p.name)

        preferred = ["Linear.pkl", "PolyDeg3.pkl", "KNN.pkl"]
        ordered = [x for x in preferred if x in candidates] + [x for x in candidates if x not in preferred]
        if not ordered:
            ordered = ["(no models found)"]

        self.model_box.addItems(ordered)
        self.model_box.blockSignals(False)

    def on_model_changed(self, name: str):
        # Load selected model pipeline and compute bias at dl=(0,0,0) for UI-friendly zero correction
        if not name or name == "(no models found)":
            self.ml_model = None
            self.ml_bias = np.zeros(3, dtype=float)
            self.update_scene()
            return

        path = MODELS_DIR / name
        try:
            self.ml_model = joblib.load(path)
            try:
                self.ml_bias = np.array(self.ml_model.predict([[0.0, 0.0, 0.0]])[0], dtype=float)
            except Exception:
                self.ml_bias = np.zeros(3, dtype=float)
        except Exception as e:
            self.ml_model = None
            self.ml_bias = np.zeros(3, dtype=float)
            QMessageBox.warning(self, "ML load error", f"Could not load model:\n{path}\n\n{e}")

        self.update_scene()

    # Core update
    def update_scene(self):
        # Read sliders and enforce max-2-tendons constraint
        l1, l2, l3 = self.dl_values()
        l1, l2, l3, clipped = self.enforce_max2(l1, l2, l3)

        if clipped:
            # Push corrected values back into sliders without triggering recursion
            self.s1.blockSignals(True); self.s2.blockSignals(True); self.s3.blockSignals(True)
            self.s1.setValue(int(round(l1 * 100)))
            self.s2.setValue(int(round(l2 * 100)))
            self.s3.setValue(int(round(l3 * 100)))
            self.s1.blockSignals(False); self.s2.blockSignals(False); self.s3.blockSignals(False)

        # Update Δl UI indicators
        self.dl1.set_value(l1)
        self.dl2.set_value(l2)
        self.dl3.set_value(l3)

        # Compute PCC centerline and tip
        Xp, Yp, Zp, theta = pcc.pcc_shape(l1, l2, l3)
        p_pcc = tip_xyz(Xp, Yp, Zp)

        # Compute REAL centerline and tip
        Xr, Yr, Zr, _ = safe_real_shape(l1, l2, l3)
        p_real = tip_xyz(Xr, Yr, Zr)

        # Predict tip correction and apply bias so dl=0 gives zero correction
        p_ml = p_pcc.copy()
        if self.ml_model is not None:
            dX, dY, dZ = self.ml_model.predict([[l1, l2, l3]])[0]
            d = np.array([dX, dY, dZ], dtype=float) - self.ml_bias
            p_ml = p_pcc + d

        # Update top cards with theta and errors (norms)
        e_pcc = float(np.linalg.norm(p_real - p_pcc))
        e_ml = float(np.linalg.norm(p_real - p_ml))
        self.card_theta.set_value(f"{np.degrees(theta):.1f}°")
        self.card_e_pcc.set_value(f"{e_pcc:.3f} mm")
        self.card_e_ml.set_value(f"{e_ml:.3f} mm")

        # Read visibility state
        show_pcc = self.cb_show_pcc.isChecked()
        show_real = self.cb_show_real.isChecked()
        show_ml = self.cb_show_ml.isChecked()

        # 3D update: draw centerlines and ML tip
        pos_pcc = np.column_stack([Xp, Yp, Zp]).astype(float)
        pos_real = np.column_stack([Xr, Yr, Zr]).astype(float)

        self.line3d_pcc.setData(pos=pos_pcc if show_pcc else np.zeros((2, 3)))
        self.line3d_real.setData(pos=pos_real if show_real else np.zeros((2, 3)))
        self.tip3d_ml.setData(pos=p_ml.reshape(1, 3) if show_ml else np.zeros((1, 3)))

        # 2D update: select projection plane and plot corresponding curves/tip
        if self.current_proj == "XZ":
            pcc_a, pcc_b = Xp, Zp
            real_a, real_b = Xr, Zr
            ml_a, ml_b = p_ml[0], p_ml[2]
            self.plot2d.setLabel("bottom", "X [mm]")
            self.plot2d.setLabel("left", "Z [mm]")
        elif self.current_proj == "XY":
            pcc_a, pcc_b = Xp, Yp
            real_a, real_b = Xr, Yr
            ml_a, ml_b = p_ml[0], p_ml[1]
            self.plot2d.setLabel("bottom", "X [mm]")
            self.plot2d.setLabel("left", "Y [mm]")
        else:
            pcc_a, pcc_b = Yp, Zp
            real_a, real_b = Yr, Zr
            ml_a, ml_b = p_ml[1], p_ml[2]
            self.plot2d.setLabel("bottom", "Y [mm]")
            self.plot2d.setLabel("left", "Z [mm]")

        self.curve2d_pcc.setData(pcc_a, pcc_b) if show_pcc else self.curve2d_pcc.setData([], [])
        self.curve2d_real.setData(real_a, real_b) if show_real else self.curve2d_real.setData([], [])
        self.tip2d_ml.setData([ml_a], [ml_b]) if show_ml else self.tip2d_ml.setData([], [])

        # Autoscale: fit visible data with a fixed margin to avoid tight clipping
        def set_range_like_matplotlib(x, y, margin=10.0):
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            if not np.any(m):
                return
            x = x[m]; y = y[m]
            xmin, xmax = float(x.min()) - margin, float(x.max()) + margin
            ymin, ymax = float(y.min()) - margin, float(y.max()) + margin
            self.plot2d.setXRange(xmin, xmax, padding=0)
            self.plot2d.setYRange(ymin, ymax, padding=0)

        xs, ys = [], []
        if show_pcc and len(pcc_a) > 0:
            xs.append(pcc_a); ys.append(pcc_b)
        if show_real and len(real_a) > 0:
            xs.append(real_a); ys.append(real_b)
        if show_ml:
            xs.append([ml_a]); ys.append([ml_b])

        if xs and ys:
            Xall = np.concatenate([np.asarray(v, float) for v in xs])
            Yall = np.concatenate([np.asarray(v, float) for v in ys])
            set_range_like_matplotlib(Xall, Yall, margin=10.0)


def try_load_fonts():
    # Load local fonts if present; app still runs fine without them
    fonts_dir = Path("fonts")
    for fp in [
        fonts_dir / "Inter.ttf",
        fonts_dir / "Inter-Regular.ttf",
        fonts_dir / "JetBrainsMono.ttf",
        fonts_dir / "JetBrainsMono-Regular.ttf",
    ]:
        if fp.exists():
            QFontDatabase.addApplicationFont(str(fp))


def apply_product_glass_style(app: QApplication):
    # Apply Fusion theme + custom stylesheet for "glass" look
    app.setStyle("Fusion")
    try_load_fonts()

    # Set a consistent base font (falls back if missing)
    app.setFont(QFont("Inter", 10))

    app.setStyleSheet(f"""
        /* Base */
                      
        QMainWindow {{
            background: #070A14;
        }}

        /* Root stays transparent so panels render cleanly */
        QWidget#Root {{
            background: transparent;
            color: rgba(235, 241, 255, 235);
            font-size: 12px;
        }}

        /* Force child widgets inside Panel/Card to stay transparent (prevents black rectangles) */
        QFrame#Panel QWidget,
        QFrame#Card QWidget {{
            background: transparent;
        }}
        QWidget {{
            background: {COL["BG"]};
            color: {COL["TEXT"]};
            font-size: 12px;
        }}
        QLabel {{
            background: transparent;
        }}

        QLabel#Hint {{
            color: rgba(150, 165, 210, 170);
            padding-top: 4px;
        }}

        /* Panels */
        QFrame#Panel {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(18, 26, 52, 205),
                stop:1 rgba(10, 16, 34, 205));
            border: 1px solid {COL["BORDER"]};
            border-radius: 18px;
        }}
        QLabel#PanelTitle {{
            color: {COL["MUTED"]};
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.3px;
            padding-left: 2px;
        }}

        /* Cards */
        QFrame#Card {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(30, 44, 90, 150),
                stop:1 rgba(14, 20, 42, 200));
            border: 1px solid {COL["BORDER"]};
            border-radius: 18px;
        }}
        QLabel#CardTitle {{
            color: {COL["MUTED"]};
            font-size: 11px;
        }}
        QLabel#MetricValue {{
            font-family: "JetBrains Mono";
            font-size: 22px;
            font-weight: 650;
            color: rgba(245, 248, 255, 255);
        }}

        /* Section labels */
        QLabel#SectionLabel {{
            color: rgba(190, 205, 235, 210);
            font-weight: 600;
            padding-top: 6px;
        }}

        /* ComboBox */
        QComboBox {{
            background: rgba(140, 160, 255, 14);
            border: 1px solid {COL["BORDER"]};
            padding: 8px 10px;
            border-radius: 12px;
        }}
        QComboBox:hover {{
            border: 1px solid {COL["BORDER_HI"]};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 26px;
        }}

        /* Segmented buttons */
        QPushButton#SegBtn {{
            background: rgba(140, 160, 255, 10);
            border: 1px solid {COL["BORDER"]};
            padding: 8px 10px;
            border-radius: 12px;
        }}
        QPushButton#SegBtn:hover {{
            border: 1px solid {COL["BORDER_HI"]};
            background: rgba(140, 160, 255, 16);
        }}
        QPushButton#SegBtn:checked {{
            background: rgba(155, 124, 255, 70);
            border: 1px solid rgba(155, 124, 255, 130);
        }}

        /* Checkboxes */
        QCheckBox {{
            background: transparent;
            spacing: 10px;
            color: rgba(225, 235, 255, 230);
        }}
        QCheckBox::indicator {{
            width: 40px;
            height: 22px;
            border-radius: 11px;
            background: rgba(140, 160, 255, 16);
            border: 1px solid {COL["BORDER"]};
        }}
        QCheckBox::indicator:checked {{
            background: rgba(155, 124, 255, 90);
            border: 1px solid rgba(155, 124, 255, 130);
        }}

        /* Δl indicators */
        QLabel#DlTitle {{
            background: transparent;
            color: rgba(200, 214, 245, 220);
            font-weight: 600;
        }}
        QFrame#DlBar {{
            min-height: 8px;
            max-height: 8px;
            background: rgba(140, 160, 255, 14);
            border-radius: 4px;
            border: 1px solid rgba(140, 160, 255, 20);
        }}
        QLabel#DlValue {{
            font-family: "JetBrains Mono";
            font-weight: 650;
        }}

        /* Sliders */
        QSlider#DlSlider::groove:horizontal {{
            background: transparent;
            height: 8px;
            background: rgba(140, 160, 255, 14);
            border-radius: 4px;
        }}
        QSlider#DlSlider::handle:horizontal {{
            width: 18px;
            height: 18px;
            margin: -5px 0;
            border-radius: 9px;
            background: rgba(155, 124, 255, 220);
            border: 1px solid rgba(255,255,255,45);
        }}
    """)

# App entry point
if __name__ == "__main__":
    # Standard Qt app bootstrap
    app = QApplication(sys.argv)
    apply_product_glass_style(app)

    # Enable antialias globally for cleaner lines
    pg.setConfigOptions(antialias=True)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())

"""Desktop dashboard for continuum-robot forward-model exploration."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea,
    QSlider, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from continuum_robot.common.paths import ROOT
from continuum_robot.gui.model_adapter import EXPERIMENTS, PHYSICAL_MODELS, PhysicalAdapter, SyntheticAdapter
from continuum_robot.gui.results_dashboard import ResultsDashboard
from continuum_robot.gui.robot_view_3d import RobotView
from continuum_robot.gui.theme import COLORS

MODEL_HEX = {"PCC": COLORS["pcc"], "KNN": COLORS["knn"], "FFNN": COLORS["ffnn"],
             "Ridge3": "#D4A76F", "Ridge2": "#B59AD6", "Linear": "#7693AF"}


def _label(text: str, object_name: str = "Muted") -> QLabel:
    label = QLabel(text)
    label.setObjectName(object_name)
    return label


class MetricCard(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("MetricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(3)
        caption = _label(title, "CardLabel")
        self.value = _label("—", "CardValue")
        self.value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(caption)
        layout.addWidget(self.value)

    def set_value(self, text: str) -> None:
        self.value.setText(text)


class ProjectionPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._cloud_source = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.views = {}
        for label, axes in (("XY", (0, 1)), ("XZ", (0, 2)), ("YZ", (1, 2))):
            view = pg.PlotWidget(title=f"{label} projection")
            view.setBackground(COLORS["surface"])
            view.showGrid(x=True, y=True, alpha=.14)
            view.setLabel("bottom", f"{label[0]} [mm]")
            view.setLabel("left", f"{label[1]} [mm]")
            view.getAxis("bottom").setTextPen(COLORS["muted"])
            view.getAxis("left").setTextPen(COLORS["muted"])
            view.getPlotItem().getViewBox().setAspectLocked(True)
            view.getPlotItem().getViewBox().setMenuEnabled(False)
            cloud = pg.ScatterPlotItem(size=2, brush=pg.mkBrush(85, 155, 190, 65), pen=None)
            view.addItem(cloud)
            pcc = view.plot(pen=pg.mkPen(COLORS["pcc"], width=2.7))
            reference = view.plot(pen=pg.mkPen(COLORS["reference"], width=2.2))
            vector = view.plot(pen=pg.mkPen(COLORS["danger"], width=1.5, style=Qt.DashLine))
            corrected = view.plot(pen=None, symbol="o", symbolSize=11,
                                  symbolBrush=COLORS["ffnn"], symbolPen=pg.mkPen("#E8FFFB", width=1))
            measured = view.plot(pen=None, symbol="o", symbolSize=9,
                                 symbolBrush=COLORS["reference"], symbolPen=pg.mkPen("#E8FFFB", width=1))
            self.views[label] = (view, axes, cloud, pcc, reference, vector, corrected, measured)
            layout.addWidget(view, 1)

    def set_scene(self, scene: dict, cloud=None, *, show_cloud=True, show_vectors=True, fit=False):
        for view, (a, b), cloud_item, pcc_item, ref_item, vector, corrected, measured in self.views.values():
            curve = scene["pcc_curve"]
            pcc_item.setData(curve[:, a], curve[:, b])
            ref = scene["reference_curve"]
            ref_item.setData(ref[:, a], ref[:, b]) if ref is not None else ref_item.setData([], [])
            tip = scene["corrected"]
            if tip is not None and not np.allclose(tip, scene["pcc"]):
                corrected.setData([tip[a]], [tip[b]], symbolBrush=MODEL_HEX.get(scene.get("display_model"), COLORS["ffnn"]))
            else:
                corrected.setData([], [])
            observation = scene["reference"]
            measured.setData([observation[a]], [observation[b]]) if observation is not None else measured.setData([], [])
            if observation is not None and show_vectors:
                vector.setData([scene["pcc"][a], observation[a]], [scene["pcc"][b], observation[b]])
            else:
                vector.setData([], [])
            if cloud is not None and show_cloud:
                if cloud is not self._cloud_source:
                    cloud_item.setData(cloud[0][:, a], cloud[0][:, b])
            else:
                cloud_item.setData([], [])
            if fit:
                points = [curve]
                if cloud is not None and show_cloud and len(cloud[0]):
                    points.append(cloud[0])
                all_points = np.vstack(points)
                xlo, xhi = all_points[:, a].min(), all_points[:, a].max()
                ylo, yhi = all_points[:, b].min(), all_points[:, b].max()
                margin = max(8., .07 * max(xhi - xlo, yhi - ylo))
                view.setXRange(float(xlo - margin), float(xhi + margin), padding=0)
                view.setYRange(float(ylo - margin), float(yhi + margin), padding=0)
        self._cloud_source = cloud if show_cloud else None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Continuum Lab · Kinematics & Error Compensation")
        self.resize(1580, 930)
        self.setMinimumSize(1050, 700)
        self.synthetic = SyntheticAdapter()
        self.physical_cache: dict[str, PhysicalAdapter] = {}
        self.adapter = self.synthetic
        self.last_context = None
        self._cloud_key = None
        self._cloud_data = None
        self._updating_controls = False
        self._build_ui()
        self._wire_shortcuts()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.setInterval(25)
        self.refresh_timer.timeout.connect(self.update_scene)
        self._select_mode()

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        page = QVBoxLayout(root)
        page.setContentsMargins(0, 0, 0, 0)
        page.setSpacing(0)

        header = QFrame()
        header.setObjectName("Header")
        bar = QHBoxLayout(header)
        bar.setContentsMargins(20, 10, 20, 10)
        bar.setSpacing(12)
        heading = QVBoxLayout()
        heading.setSpacing(1)
        heading.addWidget(_label("ROBOTICS  /  RESEARCH DASHBOARD", "Eyebrow"))
        heading.addWidget(_label("Continuum Lab", "Title"))
        bar.addLayout(heading)
        bar.addStretch(1)
        bar.addWidget(_label("MODE"))
        self.mode = QComboBox()
        self.mode.addItems(("Synthetic demonstration", "Physical measurements"))
        self.mode.setMinimumWidth(185)
        self.mode.currentIndexChanged.connect(self._select_mode)
        bar.addWidget(self.mode)
        self.experiment = QComboBox()
        self.experiment.addItems(EXPERIMENTS)
        self.experiment.setToolTip("Phase 6 v2 experimental split; its saved registration and models are loaded together.")
        self.experiment.currentTextChanged.connect(self._select_mode)
        bar.addWidget(self.experiment)
        self.model = QComboBox()
        self.model.currentTextChanged.connect(self.schedule_refresh)
        self.model.setMinimumWidth(105)
        bar.addWidget(self.model)
        self.dataset_status = _label("", "Status")
        self.dataset_status.setMinimumWidth(190)
        bar.addWidget(self.dataset_status)
        fit_button = QPushButton("Fit view")
        fit_button.clicked.connect(self.fit_view)
        fit_button.setToolTip("Recenter the 3D view and projections (F)")
        bar.addWidget(fit_button)
        shot = QPushButton("Capture")
        shot.setObjectName("Primary")
        shot.clicked.connect(self.capture_screenshot)
        shot.setToolTip("Save an actual dashboard screenshot (Ctrl+Shift+S)")
        bar.addWidget(shot)
        page.addWidget(header)

        content = QWidget()
        body = QVBoxLayout(content)
        body.setContentsMargins(14, 12, 14, 10)
        body.setSpacing(10)
        cards = QGridLayout()
        cards.setSpacing(9)
        definitions = (("BEND ANGLE", "theta"), ("REGISTERED PCC · XYZ", "pcc"),
                       ("CORRECTED TIP · XYZ", "corrected"), ("REFERENCE / MEASURED · XYZ", "reference"),
                       ("PCC ERROR", "pcc_error"), ("CORRECTED ERROR", "corrected_error"),
                       ("ERROR CHANGE", "improvement"), ("DATA SOURCE", "source"))
        self.cards = {}
        for i, (title, key) in enumerate(definitions):
            card = MetricCard(title)
            self.cards[key] = card
            cards.addWidget(card, i // 4, i % 4)
        body.addLayout(cards)

        horizontal = QSplitter(Qt.Horizontal)
        horizontal.setChildrenCollapsible(False)
        visual = QSplitter(Qt.Vertical)
        visual.setChildrenCollapsible(False)
        viewport_panel = QFrame()
        viewport_panel.setObjectName("Panel")
        viewport_layout = QVBoxLayout(viewport_panel)
        viewport_layout.setContentsMargins(10, 9, 10, 8)
        viewport_layout.setSpacing(6)
        viewport_title = QHBoxLayout()
        viewport_title.addWidget(_label("3D MODEL VIEWPORT", "SectionTitle"))
        viewport_title.addStretch(1)
        viewport_title.addWidget(_label("X  ● red    Y  ● green    Z  ● blue     ·     drag rotate / wheel zoom / middle drag pan"))
        viewport_layout.addLayout(viewport_title)
        self.view3d = RobotView()
        viewport_layout.addWidget(self.view3d, 1)
        legend = _label("●  PCC backbone     ●  Synthetic reference / measured endpoint     ●  Corrected endpoint     ·  Tube radius is illustrative")
        legend.setWordWrap(True)
        viewport_layout.addWidget(legend)
        visual.addWidget(viewport_panel)

        self.tabs = QTabWidget()
        self.projections = ProjectionPanel()
        self.tabs.addTab(self.projections, "XY / XZ / YZ")
        self.results = ResultsDashboard()
        self.tabs.addTab(self.results, "Experimental results")
        visual.addWidget(self.tabs)
        visual.setSizes([580, 290])
        horizontal.addWidget(visual)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(285)
        controls_scroll.setMaximumWidth(385)
        panel = QFrame()
        panel.setObjectName("Panel")
        controls_scroll.setWidget(panel)
        controls = QVBoxLayout(panel)
        controls.setContentsMargins(16, 14, 16, 14)
        controls.setSpacing(12)
        controls.addWidget(_label("ACTUATOR COMMANDS", "Eyebrow"))
        controls.addWidget(_label("Tendon shortening · 0–10 mm", "SectionTitle"))
        self.tendon_sliders = []
        self.tendon_inputs = []
        self.tendon_labels = []
        for index in range(3):
            row = QHBoxLayout()
            name = _label(f"L{index + 1}", "SectionTitle")
            name.setMinimumWidth(24)
            row.addWidget(name)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 1000)
            slider.setToolTip(f"Tendon {index + 1} shortening. At most two tendons may be active.")
            row.addWidget(slider, 1)
            numeric = QDoubleSpinBox()
            numeric.setRange(0, 10)
            numeric.setDecimals(2)
            numeric.setSingleStep(.1)
            numeric.setSuffix(" mm")
            numeric.setMinimumWidth(95)
            row.addWidget(numeric)
            slider.valueChanged.connect(lambda value, i=index: self._slider_changed(i, value))
            numeric.valueChanged.connect(lambda value, i=index: self._number_changed(i, value))
            self.tendon_sliders.append(slider)
            self.tendon_inputs.append(numeric)
            self.tendon_labels.append(name)
            controls.addLayout(row)
        self.constraint = _label("At most two active tendons · constraint enforced")
        self.constraint.setWordWrap(True)
        controls.addWidget(self.constraint)
        presets = QGridLayout()
        for i, (title, values) in enumerate((("Straight", (0, 0, 0)), ("Plane 1", (5, 5, 0)),
                                             ("Plane 2", (5, 0, 5)), ("Plane 3", (0, 5, 5)))):
            button = QPushButton(title)
            button.clicked.connect(lambda _, v=values: self.set_tendons(v))
            presets.addWidget(button, i // 2, i % 2)
        controls.addLayout(presets)
        controls.addWidget(_label("CAMERA", "Eyebrow"))
        cameras = QGridLayout()
        for i, name in enumerate(("Isometric", "Top", "Front", "Side")):
            button = QPushButton(name)
            button.clicked.connect(lambda _, v=name: self.view3d.camera_preset(v))
            cameras.addWidget(button, i // 2, i % 2)
        controls.addLayout(cameras)
        self.show_cloud = QCheckBox("Measured test workspace cloud")
        self.show_cloud.setChecked(True)
        self.show_cloud.stateChanged.connect(self.schedule_refresh)
        controls.addWidget(self.show_cloud)
        self.show_vectors = QCheckBox("Endpoint error vectors")
        self.show_vectors.setChecked(True)
        self.show_vectors.stateChanged.connect(self.schedule_refresh)
        controls.addWidget(self.show_vectors)
        controls.addWidget(_label("TEST OBSERVATION", "Eyebrow"))
        self.point_index = QSpinBox()
        self.point_index.setMinimum(0)
        self.point_index.setToolTip("Jump to an actual endpoint measurement in the selected Phase 6 v2 test split.")
        controls.addWidget(self.point_index)
        jump = QPushButton("Go to test point")
        jump.clicked.connect(self.jump_to_test_point)
        controls.addWidget(jump)
        self.detail = _label("Physical mode shows only test-set observed endpoints. Corrected models predict tip position, not backbone shape.")
        self.detail.setWordWrap(True)
        controls.addWidget(self.detail)
        controls.addStretch(1)
        horizontal.addWidget(controls_scroll)
        horizontal.setSizes([1160, 345])
        body.addWidget(horizontal, 1)
        page.addWidget(content, 1)
        self.statusBar().showMessage("Ready · no experimental code runs from this dashboard")

    def _wire_shortcuts(self):
        QShortcut(QKeySequence("R"), self, activated=lambda: self.set_tendons((0, 0, 0)))
        QShortcut(QKeySequence("F"), self, activated=self.fit_view)
        QShortcut(QKeySequence("Ctrl+Shift+S"), self, activated=self.capture_screenshot)

    def _slider_changed(self, index: int, value: int):
        if self._updating_controls:
            return
        with QSignalBlocker(self.tendon_inputs[index]):
            self.tendon_inputs[index].setValue(value / 100)
        self._enforce_constraint(index)
        self.schedule_refresh()

    def _number_changed(self, index: int, value: float):
        if self._updating_controls:
            return
        with QSignalBlocker(self.tendon_sliders[index]):
            self.tendon_sliders[index].setValue(round(value * 100))
        self._enforce_constraint(index)
        self.schedule_refresh()

    def _enforce_constraint(self, changed: int):
        active = [i for i, widget in enumerate(self.tendon_inputs) if widget.value() > 0]
        if len(active) <= 2:
            return
        other = min((i for i in active if i != changed), key=lambda i: self.tendon_inputs[i].value())
        self._set_one(other, 0)
        self.statusBar().showMessage(f"Tendon {other + 1} reset to keep at most two active", 4000)

    def _set_one(self, index: int, value: float):
        with QSignalBlocker(self.tendon_inputs[index]), QSignalBlocker(self.tendon_sliders[index]):
            self.tendon_inputs[index].setValue(value)
            self.tendon_sliders[index].setValue(round(value * 100))

    def set_tendons(self, values):
        self._updating_controls = True
        try:
            for i, value in enumerate(values):
                self._set_one(i, float(value))
        finally:
            self._updating_controls = False
        self.schedule_refresh()

    def schedule_refresh(self):
        if hasattr(self, "refresh_timer"):
            self.refresh_timer.start()

    def _select_mode(self):
        physical = self.mode.currentIndex() == 1
        self.experiment.setVisible(physical)
        self.point_index.setEnabled(physical)
        self.show_cloud.setEnabled(physical)
        with QSignalBlocker(self.model):
            self.model.clear()
            self.model.addItems(PHYSICAL_MODELS if physical else ("PCC + FFNN",))
        if not physical:
            self.adapter = self.synthetic
            self.dataset_status.setText("SYNTHETIC · model reference")
            self.results.clear("Phase 6 results are available in Physical measurements mode.")
            self.detail.setText("Synthetic reference is deterministic model output. Noise is disabled for stable exploration; the separate tip offset remains included.")
        else:
            name = self.experiment.currentText()
            try:
                if name not in self.physical_cache:
                    self.physical_cache[name] = PhysicalAdapter(name)
                self.adapter = self.physical_cache[name]
                self.point_index.setMaximum(len(self.adapter.predictions) - 1)
                self.point_index.setValue(0)
                self.dataset_status.setText(f"PHYSICAL TEST · {name}")
                self.results.show_experiment(self.adapter)
                self.detail.setText("Cloud and measurements are physical TEST endpoints only. The tube is registered PCC geometry; corrected models provide endpoint predictions only.")
                self.set_tendons(self.adapter.test_command(0))
            except Exception as exc:
                self.adapter = None
                self.dataset_status.setText("ARTIFACTS UNAVAILABLE")
                self.results.clear(str(exc))
                self.detail.setText(str(exc))
                self.statusBar().showMessage(str(exc))
        self.last_context = None
        self.schedule_refresh()

    def jump_to_test_point(self):
        if isinstance(self.adapter, PhysicalAdapter):
            self.set_tendons(self.adapter.test_command(self.point_index.value()))

    def _format_xyz(self, value):
        return "—" if value is None else f"X {value[0]:.1f}  Y {value[1]:.1f}\nZ {value[2]:.1f} mm"

    def update_scene(self):
        if self.adapter is None:
            for key in self.cards:
                self.cards[key].set_value("—")
            return
        dl = np.array([widget.value() for widget in self.tendon_inputs], float)
        for i, label in enumerate(self.tendon_labels):
            label.setStyleSheet(f"color: {COLORS['accent'] if dl[i] > 0 else COLORS['muted']};")
        physical = isinstance(self.adapter, PhysicalAdapter)
        model_name = self.model.currentText() if physical else "FFNN"
        try:
            scene = self.adapter.evaluate(dl, model_name) if physical else self.adapter.evaluate(dl)
        except Exception as exc:
            self.statusBar().showMessage(f"Model unavailable: {exc}")
            self.dataset_status.setText("MODEL UNAVAILABLE")
            return
        scene["display_model"] = model_name
        self.dataset_status.setText(f"PHYSICAL TEST · {self.adapter.experiment}" if physical
                                    else "SYNTHETIC · model reference")
        cloud_key = (id(self.adapter), model_name) if physical else None
        if cloud_key != self._cloud_key:
            self._cloud_data = self.adapter.cloud(model_name) if physical else None
            self._cloud_key = cloud_key
        cloud = self._cloud_data
        context = ("physical", self.adapter.experiment) if physical else ("synthetic",)
        changed_context = context != self.last_context
        self.view3d.set_scene(scene, cloud=cloud, show_cloud=self.show_cloud.isChecked() and physical,
                              show_vectors=self.show_vectors.isChecked())
        self.projections.set_scene(scene, cloud=cloud, show_cloud=self.show_cloud.isChecked() and physical,
                                   show_vectors=self.show_vectors.isChecked(), fit=changed_context)
        if changed_context:
            self.view3d.fit_view()
        self.last_context = context
        pcc_error = None if scene["reference"] is None else float(np.linalg.norm(scene["pcc"] - scene["reference"]))
        corrected_error = None if scene["reference"] is None or scene["corrected"] is None else float(np.linalg.norm(scene["corrected"] - scene["reference"]))
        self.cards["theta"].set_value(f"{scene['theta_deg']:.1f}°")
        self.cards["pcc"].set_value(self._format_xyz(scene["pcc"]))
        self.cards["corrected"].set_value(self._format_xyz(scene["corrected"]))
        self.cards["reference"].set_value(self._format_xyz(scene["reference"]))
        self.cards["pcc_error"].set_value("—" if pcc_error is None else f"{pcc_error:.3f} mm")
        self.cards["corrected_error"].set_value("—" if corrected_error is None else f"{corrected_error:.3f} mm")
        if corrected_error is not None and pcc_error is not None and pcc_error > 1e-9:
            change = 100 * (1 - corrected_error / pcc_error)
            self.cards["improvement"].set_value(f"{change:+.1f}%")
        else:
            self.cards["improvement"].set_value("—")
        self.cards["source"].set_value("Physical · v2 test" if physical else "Synthetic · deterministic")
        self.statusBar().showMessage(scene["reference_label"] + " · " + scene["frame_label"] +
                                     (" · " + scene["warning"] if scene["warning"] else ""))

    def fit_view(self):
        self.view3d.fit_view()
        self.last_context = None
        self.schedule_refresh()

    def capture_screenshot(self):
        destination = ROOT / "figures" / "gui"
        destination.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, "Capture actual dashboard", str(destination / "dashboard_overview.png"),
                                              "PNG image (*.png)")
        if path:
            if self.grab().save(path, "PNG"):
                self.statusBar().showMessage(f"Screenshot saved: {path}", 7000)
            else:
                self.statusBar().showMessage(f"Could not save screenshot: {path}", 7000)

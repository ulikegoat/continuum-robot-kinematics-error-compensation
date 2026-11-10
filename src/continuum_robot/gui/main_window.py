"""Desktop dashboard for continuum-robot forward-model exploration."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPoint, QRect, QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea,
    QSlider, QSpinBox, QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from continuum_robot.common.paths import ROOT
from continuum_robot.gui.model_adapter import EXPERIMENTS, PHYSICAL_MODELS, PhysicalAdapter, SyntheticAdapter
from continuum_robot.gui.results_dashboard import ResultsDashboard
from continuum_robot.gui.robot_view_3d import RobotView
from continuum_robot.gui.theme import COLORS

MODEL_HEX = {name: COLORS[key] for name, key in (
    ("PCC", "pcc"), ("KNN", "knn"), ("FFNN", "ffnn"),
    ("Ridge3", "ridge3"), ("Ridge2", "ridge2"), ("Linear", "linear"))}


def _label(text: str, object_name: str = "Muted") -> QLabel:
    label = QLabel(text)
    label.setObjectName(object_name)
    return label


class TelemetryCell(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("TelemetryCell")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(9, 4, 9, 4)
        layout.setSpacing(1)
        self.caption = _label(title, "CardLabel")
        self.value = _label("—", "CardValue")
        self.value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.caption)
        layout.addWidget(self.value)

    def set_value(self, text: str) -> None:
        self.value.setText(text)

    def set_title(self, text: str) -> None:
        self.caption.setText(text)


class InspectorGroup(QWidget):
    """Compact collapsible control section used by the right inspector."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.header = QPushButton(f"▾  {title}")
        self.header.setObjectName("Group")
        self.header.setCheckable(True)
        self.header.setChecked(True)
        self.header.toggled.connect(self._toggle)
        outer.addWidget(self.header)
        self.body = QWidget()
        self.layout = QVBoxLayout(self.body)
        self.layout.setContentsMargins(3, 7, 3, 9)
        self.layout.setSpacing(8)
        outer.addWidget(self.body)

    def _toggle(self, expanded: bool):
        self.header.setText(f"{'▾' if expanded else '▸'}  {self.header.text()[3:]}")
        self.body.setVisible(expanded)


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
            view.getAxis("bottom").setPen(COLORS["border"])
            view.getAxis("left").setPen(COLORS["border"])
            view.getPlotItem().getViewBox().setAspectLocked(True)
            view.getPlotItem().getViewBox().setMenuEnabled(False)
            cloud = pg.ScatterPlotItem(size=2, brush=pg.mkBrush(130, 123, 110, 65), pen=None)
            view.addItem(cloud)
            pcc = view.plot(pen=pg.mkPen(COLORS["pcc"], width=2.7))
            reference = view.plot(pen=pg.mkPen(COLORS["reference"], width=2.2))
            vector = view.plot(pen=pg.mkPen(COLORS["danger"], width=1.5, style=Qt.DashLine))
            corrected = view.plot(pen=None, symbol="o", symbolSize=11,
                                  symbolBrush=COLORS["ffnn"], symbolPen=pg.mkPen(COLORS["surface"], width=1))
            measured = view.plot(pen=None, symbol="o", symbolSize=9,
                                 symbolBrush=COLORS["reference"], symbolPen=pg.mkPen(COLORS["surface"], width=1))
            self.views[label] = (view, axes, cloud, pcc, reference, vector, corrected, measured)
            layout.addWidget(view, 1)

    def set_scene(self, scene: dict, cloud=None, *, show_cloud=True, show_vectors=True,
                  fit=False, fit_workspace=False):
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
                for endpoint in (scene["reference"], scene["corrected"]):
                    if endpoint is not None:
                        points.append(np.asarray(endpoint, float).reshape(1, 3))
                if fit_workspace and cloud is not None and show_cloud and len(cloud[0]):
                    points.append(cloud[0])
                all_points = np.vstack(points)
                xlo, xhi = all_points[:, a].min(), all_points[:, a].max()
                ylo, yhi = all_points[:, b].min(), all_points[:, b].max()
                margin = max(8., .07 * max(xhi - xlo, yhi - ylo))
                view.setXRange(float(xlo - margin), float(xhi + margin), padding=0)
                view.setYRange(float(ylo - margin), float(yhi + margin), padding=0)
        self._cloud_source = cloud if show_cloud else None

    def clear_scene(self):
        for view, axes, cloud, pcc, reference, vector, corrected, measured in self.views.values():
            for item in (cloud, pcc, reference, vector, corrected, measured):
                item.setData([], [])
        self._cloud_source = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Continuum Lab · Kinematics & Error Compensation")
        self.resize(1580, 930)
        self.setMinimumSize(960, 650)
        self.synthetic = SyntheticAdapter()
        self.physical_cache: dict[str, PhysicalAdapter] = {}
        self.adapter = self.synthetic
        self.last_context = None
        self._cloud_key = None
        self._cloud_data = None
        self._projection_cloud_data = None
        self._projection_frame = "robot"
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
        header.setObjectName("Toolbar")
        header.setFixedHeight(49)
        bar = QHBoxLayout(header)
        bar.setContentsMargins(12, 4, 12, 4)
        bar.setSpacing(8)
        bar.addWidget(_label("◧  Continuum Lab", "Title"))
        bar.addStretch(1)
        self.mode = QComboBox()
        self.mode.addItems(("Synthetic demonstration", "Physical measurements"))
        self.mode.setMinimumWidth(160)
        self.mode.setToolTip("Data source: simulated reference or saved physical endpoints")
        self.mode.currentIndexChanged.connect(self._select_mode)
        bar.addWidget(self.mode)
        self.experiment = QComboBox()
        self.experiment.addItems(EXPERIMENTS)
        self.experiment.setMinimumWidth(100)
        self.experiment.setToolTip("Phase 6 v2 experimental split; its saved registration and models are loaded together.")
        self.experiment.currentTextChanged.connect(self._select_mode)
        bar.addWidget(self.experiment)
        self.model = QComboBox()
        self.model.currentTextChanged.connect(self.schedule_refresh)
        self.model.setMinimumWidth(105)
        bar.addWidget(self.model)
        shot = QPushButton("Capture")
        shot.setObjectName("Primary")
        shot.clicked.connect(self.capture_screenshot)
        shot.setToolTip("Save an actual dashboard screenshot (Ctrl+Shift+S)")
        bar.addWidget(shot)
        page.addWidget(header)

        content = QWidget()
        body = QHBoxLayout(content)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        nav = QFrame()
        nav.setObjectName("Navigation")
        nav.setFixedWidth(108)
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(0, 12, 0, 4)
        nav_layout.setSpacing(2)
        self.nav_buttons = []
        for index, name in enumerate(("ROBOT", "COMPENSATION", "MEASUREMENTS", "ANALYSIS")):
            button = QPushButton(name.title())
            button.setObjectName("Nav")
            button.setCheckable(True)
            button.setToolTip(f"Open the {name.lower()} workspace")
            button.clicked.connect(lambda _, i=index: self._choose_workspace(i))
            nav_layout.addWidget(button)
            self.nav_buttons.append(button)
        nav_layout.addStretch(1)
        self.nav_buttons[0].setChecked(True)
        body.addWidget(nav)
        self.workspace_stack = QStackedWidget()
        body.addWidget(self.workspace_stack, 1)
        cards = QGridLayout()
        cards.setSpacing(0)
        definitions = (("Bend angle", "theta"), ("PCC tip · XYZ", "pcc"),
                       ("Selected model · XYZ", "corrected"), ("Reference / measured · XYZ", "reference"),
                       ("PCC point error", "pcc_error"), ("Selected point error", "corrected_error"),
                       ("Error change", "improvement"), ("Data source", "source"))
        self.cards = {}
        for i, (title, key) in enumerate(definitions):
            card = TelemetryCell(title)
            self.cards[key] = card
            cards.addWidget(card, i // 4, i % 4)
        for key in ("pcc_error", "corrected_error", "improvement"):
            self.cards[key].setToolTip("Current displayed endpoint only. Aggregate test RMSE is in Experimental results.")
        telemetry = QFrame()
        telemetry.setObjectName("Telemetry")
        telemetry.setLayout(cards)

        visual = QSplitter(Qt.Vertical)
        visual.setChildrenCollapsible(False)
        viewport_panel = QFrame()
        viewport_panel.setObjectName("Panel")
        viewport_layout = QVBoxLayout(viewport_panel)
        viewport_layout.setContentsMargins(7, 7, 7, 5)
        viewport_layout.setSpacing(4)
        viewport_title = QHBoxLayout()
        viewport_title.setSpacing(4)
        self.workspace_caption = _label("ROBOT GEOMETRY  /  XYZ [mm]", "SectionTitle")
        viewport_title.addWidget(self.workspace_caption)
        viewport_title.addStretch(1)
        for title, action, hint in (("ISO", self.reset_camera, "Reset isometric camera"),
                                    ("TOP", lambda: self.view3d.camera_preset("Top"), "Top view"),
                                    ("FIT", self.focus_robot, "Focus selected robot (F)"),
                                    ("ALL", self.fit_workspace, "Fit full workspace"),
                                    ("CLOUD", self._cycle_cloud, "Cycle Off / Light / Full cloud"),
                                    ("ERR", self._toggle_vectors, "Toggle endpoint error vectors"),
                                    ("PNG", self.capture_viewport, "Save 3D viewport PNG")):
            button = QPushButton(title)
            button.setObjectName("ViewportTool")
            button.setToolTip(hint)
            button.clicked.connect(action)
            viewport_title.addWidget(button)
            if title == "CLOUD":
                self.cloud_button = button
            elif title == "ERR":
                self.vector_button = button
        viewport_layout.addLayout(viewport_title)
        self.view3d = RobotView()
        viewport_layout.addWidget(self.view3d, 1)
        self.viewport_legend = _label("")
        self.viewport_legend.setWordWrap(True)
        viewport_layout.addWidget(self.viewport_legend)
        visual.addWidget(viewport_panel)
        self.projections = ProjectionPanel()
        visual.addWidget(self.projections)
        visual.setSizes([690, 210])
        explorer = QWidget()
        explorer_layout = QVBoxLayout(explorer)
        explorer_layout.setContentsMargins(8, 8, 8, 6)
        explorer_layout.setSpacing(6)
        explorer_layout.addWidget(visual, 1)
        cards.setContentsMargins(0, 0, 0, 0)
        explorer_layout.addWidget(telemetry)
        self.workspace_stack.addWidget(explorer)

        analysis = QWidget()
        analysis_layout = QVBoxLayout(analysis)
        analysis_layout.setContentsMargins(10, 10, 10, 10)
        analysis_layout.setSpacing(6)
        analysis_layout.addWidget(_label("PHASE 6  /  SAVED EXPERIMENT ANALYSIS", "SectionTitle"))
        self.results = ResultsDashboard()
        analysis_layout.addWidget(self.results, 1)
        self.workspace_stack.addWidget(analysis)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFixedWidth(306)
        panel = QFrame()
        panel.setObjectName("Inspector")
        controls_scroll.setWidget(panel)
        controls = QVBoxLayout(panel)
        controls.setContentsMargins(12, 10, 12, 8)
        controls.setSpacing(4)
        controls.addWidget(_label("INSPECTOR  /  ACTIVE CONFIGURATION", "SectionTitle"))
        actuation_group = InspectorGroup("ACTUATION")
        controls.addWidget(actuation_group)
        actuation = actuation_group.layout
        actuation.addWidget(_label("Tendon shortening · 0–10 mm"))
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
            actuation.addLayout(row)
        self.constraint = _label("At most two active tendons · constraint enforced")
        self.constraint.setWordWrap(True)
        actuation.addWidget(self.constraint)
        presets = QGridLayout()
        for i, (title, values) in enumerate((("Straight", (0, 0, 0)), ("Plane 1", (5, 5, 0)),
                                             ("Plane 2", (5, 0, 5)), ("Plane 3", (0, 5, 5)))):
            button = QPushButton(title)
            button.clicked.connect(lambda _, v=values: self.set_tendons(v))
            presets.addWidget(button, i // 2, i % 2)
        actuation.addLayout(presets)

        model_group = InspectorGroup("MODEL")
        controls.addWidget(model_group)
        model_group.layout.addWidget(_label("Prediction source and saved experiment"))
        self.dataset_status = _label("", "Status")
        self.dataset_status.setWordWrap(True)
        model_group.layout.addWidget(self.dataset_status)
        self.inspector_model = _label("", "SectionTitle")
        self.inspector_model.setWordWrap(True)
        model_group.layout.addWidget(self.inspector_model)

        display_group = InspectorGroup("DISPLAY")
        controls.addWidget(display_group)
        display = display_group.layout
        display.addWidget(_label("Camera presets"))
        cameras = QGridLayout()
        for i, name in enumerate(("Isometric", "Top", "Front", "Side")):
            button = QPushButton(name)
            button.clicked.connect(lambda _, v=name: self.view3d.camera_preset(v))
            cameras.addWidget(button, i // 2, i % 2)
        display.addLayout(cameras)
        reset_camera = QPushButton("Reset camera · isometric")
        reset_camera.setToolTip("Restore the default isometric angle and frame the selected robot.")
        reset_camera.clicked.connect(self.reset_camera)
        display.addWidget(reset_camera)
        framing = QHBoxLayout()
        focus = QPushButton("Focus robot")
        focus.clicked.connect(self.focus_robot)
        workspace = QPushButton("Fit workspace")
        workspace.clicked.connect(self.fit_workspace)
        framing.addWidget(focus)
        framing.addWidget(workspace)
        display.addLayout(framing)
        viewport_capture = QPushButton("Save 3D viewport PNG")
        viewport_capture.setToolTip("Capture the actual OpenGL framebuffer without desktop elements.")
        viewport_capture.clicked.connect(self.capture_viewport)
        display.addWidget(viewport_capture)
        self.show_robot = QCheckBox("Analytical backbone (3D)")
        self.show_robot.setChecked(True)
        self.show_robot.stateChanged.connect(self.schedule_refresh)
        display.addWidget(self.show_robot)
        display.addWidget(_label("Measured test cloud"))
        self.cloud_mode = QComboBox()
        self.cloud_mode.addItems(("Off", "Light", "Full"))
        self.cloud_mode.setCurrentText("Light")
        self.cloud_mode.setToolTip("Light samples the test cloud for robot-focused exploration. Full shows every test endpoint.")
        self.cloud_mode.currentTextChanged.connect(self._cloud_changed)
        display.addWidget(self.cloud_mode)
        self.cloud_scale = _label("Error colour: gray → copper · selected model")
        self.cloud_scale.setWordWrap(True)
        display.addWidget(self.cloud_scale)
        self.show_vectors = QCheckBox("Endpoint error vectors")
        self.show_vectors.setChecked(True)
        self.show_vectors.stateChanged.connect(self._vectors_changed)
        display.addWidget(self.show_vectors)

        selection_group = InspectorGroup("SELECTION")
        controls.addWidget(selection_group)
        selection = selection_group.layout
        selection.addWidget(_label("Physical test observation index"))
        self.point_index = QSpinBox()
        self.point_index.setMinimum(0)
        self.point_index.setToolTip("Jump to an actual endpoint measurement in the selected Phase 6 v2 test split.")
        selection.addWidget(self.point_index)
        self.jump_button = QPushButton("Go to test point")
        self.jump_button.clicked.connect(self.jump_to_test_point)
        selection.addWidget(self.jump_button)
        self.selection_values = {}
        for label, key in (("Measured / reference", "reference"),
                           ("PCC prediction", "pcc"), ("Selected prediction", "corrected")):
            selection.addWidget(_label(label))
            value = _label("—", "CardValue")
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            selection.addWidget(value)
            self.selection_values[key] = value
        self.detail = _label("Physical mode shows only test-set observed endpoints. Corrected models predict tip position, not backbone shape.")
        self.detail.setWordWrap(True)
        selection.addWidget(self.detail)
        controls.addStretch(1)
        body.addWidget(controls_scroll)
        page.addWidget(content, 1)
        self._cloud_changed()
        self._vectors_changed()
        self.statusBar().showMessage("Ready · no experimental code runs from this dashboard")

    def _wire_shortcuts(self):
        QShortcut(QKeySequence("R"), self, activated=lambda: self.set_tendons((0, 0, 0)))
        QShortcut(QKeySequence("F"), self, activated=self.focus_robot)
        QShortcut(QKeySequence("Ctrl+Shift+S"), self, activated=self.capture_screenshot)

    def _choose_workspace(self, index: int):
        for i, button in enumerate(self.nav_buttons):
            button.setChecked(i == index)
        self.workspace_stack.setCurrentIndex(1 if index == 3 else 0)
        if index < 3:
            self.workspace_caption.setText(("ROBOT GEOMETRY", "MODEL COMPENSATION", "MEASURED WORKSPACE")[index]
                                           + "  /  XYZ [mm]")
        self.mode.setCurrentIndex(0 if index == 0 else 1)
        if index == 2 and self.cloud_mode.currentText() == "Off":
            self.cloud_mode.setCurrentText("Light")
        if index == 2:
            self.fit_workspace()
        elif index in (0, 1):
            self.focus_robot()

    def _cycle_cloud(self):
        if not self.cloud_mode.isEnabled():
            return
        self.cloud_mode.setCurrentIndex((self.cloud_mode.currentIndex() + 1) % self.cloud_mode.count())

    def _toggle_vectors(self):
        self.show_vectors.setChecked(not self.show_vectors.isChecked())

    def _cloud_changed(self):
        mode = self.cloud_mode.currentText()
        self.cloud_button.setText(f"CLD:{mode[0]}")
        self.cloud_button.setToolTip(f"Cloud: {mode}. Click to cycle Off / Light / Full.")
        self.schedule_refresh()

    def _vectors_changed(self):
        on = self.show_vectors.isChecked()
        self.vector_button.setText("ERR:ON" if on else "ERR:OFF")
        self.vector_button.setToolTip("Hide endpoint error vectors" if on else "Show endpoint error vectors")
        self.schedule_refresh()

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
        self.jump_button.setEnabled(physical)
        self.cloud_mode.setEnabled(physical)
        self.cloud_button.setEnabled(physical)
        with QSignalBlocker(self.model):
            self.model.clear()
            self.model.addItems(PHYSICAL_MODELS if physical else ("PCC + FFNN",))
        if not physical:
            self.adapter = self.synthetic
            self.dataset_status.setText("SYNTHETIC · model reference")
            self.dataset_status.setStyleSheet(f"color: {COLORS['positive']};")
            self.results.clear("Phase 6 results are available in Physical measurements mode.")
            self.detail.setText("Synthetic reference is deterministic model output. Noise is disabled for stable exploration; the separate tip offset remains included.")
        else:
            name = self.experiment.currentText()
            try:
                if name not in self.physical_cache:
                    self.physical_cache[name] = PhysicalAdapter(name)
                self.adapter = self.physical_cache[name]
                preferred = self.adapter.metrics.set_index("method").loc[["KNN", "FFNN"], "RMSE_norm"].idxmin()
                with QSignalBlocker(self.model):
                    self.model.setCurrentText(preferred)
                self.point_index.setMaximum(len(self.adapter.predictions) - 1)
                self.point_index.setValue(0)
                self.dataset_status.setText(f"PHYSICAL TEST · {name}")
                self.dataset_status.setStyleSheet(f"color: {COLORS['positive']};")
                self.results.show_experiment(self.adapter)
                self.detail.setText("Cloud and measurements are physical TEST endpoints only. The tube is registered PCC geometry; corrected models provide endpoint predictions only.")
                self.set_tendons(self.adapter.test_command(0))
            except Exception as exc:
                self.adapter = None
                self.dataset_status.setText("ARTIFACTS UNAVAILABLE")
                self.dataset_status.setStyleSheet(f"color: {COLORS['danger']};")
                self.results.clear(str(exc))
                self.detail.setText(str(exc))
                self.statusBar().showMessage(str(exc))
        self.last_context = None
        self._projection_frame = "robot"
        self.schedule_refresh()

    def jump_to_test_point(self):
        if isinstance(self.adapter, PhysicalAdapter):
            self.set_tendons(self.adapter.test_command(self.point_index.value()))

    def _format_xyz(self, value, *, empty="Unavailable"):
        return empty if value is None else f"X {value[0]:.2f}  Y {value[1]:.2f}\nZ {value[2]:.2f} mm"

    def update_scene(self):
        if self.adapter is None:
            for key in self.cards:
                self.cards[key].set_value("—")
            for value in self.selection_values.values():
                value.setText("—")
            self.inspector_model.setText("Model artifacts unavailable")
            self.view3d.clear_scene()
            self.projections.clear_scene()
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
            self.dataset_status.setStyleSheet(f"color: {COLORS['danger']};")
            self.view3d.clear_scene()
            self.projections.clear_scene()
            for key in self.cards:
                self.cards[key].set_value("—")
            for value in self.selection_values.values():
                value.setText("—")
            self.inspector_model.setText(f"{model_name} unavailable")
            return
        scene["display_model"] = model_name
        model_color = MODEL_HEX.get(model_name, COLORS["ffnn"])
        self.viewport_legend.setText(
            f"<span style='color:{COLORS['pcc']}'>●</span> PCC &nbsp; "
            f"<span style='color:{COLORS['reference']}'>●</span> {'Measured' if physical else 'Synthetic reference'} &nbsp; "
            f"<span style='color:{model_color}'>●</span> {model_name} tip &nbsp; "
            "| drag rotate · wheel zoom · middle drag pan · tube radius illustrative")
        if physical:
            model_context = "registered PCC + residual" if model_name != "PCC" else "registered PCC baseline"
        else:
            model_context = "Phase 3 NN · synthetic reference"
        self.inspector_model.setText(f"{model_name} · {model_context}")
        self.dataset_status.setText(f"PHYSICAL TEST · {self.adapter.experiment}" if physical
                                    else "SYNTHETIC · model reference")
        cloud_mode = self.cloud_mode.currentText() if physical else "Off"
        cloud_key = (id(self.adapter), model_name, cloud_mode) if physical else None
        if cloud_key != self._cloud_key:
            self._cloud_data = (self.adapter.cloud(model_name, limit=None if cloud_mode == "Full" else 750)
                                if physical and cloud_mode != "Off" else None)
            self._projection_cloud_data = (self.adapter.cloud(model_name, limit=1800)
                                           if physical and cloud_mode == "Full" else self._cloud_data)
            self._cloud_key = cloud_key
        cloud = self._cloud_data
        scale = (float(np.percentile(self.adapter.predictions[f"e_{model_name}"], 95))
                 if physical and cloud_mode != "Off" else None)
        self.cloud_scale.setText((f"Gray: 0 mm    →    Copper: ≥{scale:.2f} mm (P95) · "
                                  f"{len(cloud[0]):,} test points displayed" if scale is not None and cloud is not None
                                  else "Cloud hidden · selected endpoint remains visible"))
        context = ("physical", self.adapter.experiment) if physical else ("synthetic",)
        changed_context = context != self.last_context
        self.view3d.set_scene(scene, cloud=cloud, cloud_mode=cloud_mode,
                              show_vectors=self.show_vectors.isChecked(),
                              show_robot=self.show_robot.isChecked(), error_scale_mm=scale)
        self.projections.set_scene(scene, cloud=self._projection_cloud_data, show_cloud=cloud_mode != "Off",
                                   show_vectors=self.show_vectors.isChecked(), fit=changed_context,
                                   fit_workspace=self._projection_frame == "workspace")
        if changed_context:
            if self._projection_frame == "workspace":
                self.view3d.fit_workspace()
            else:
                self.view3d.focus_robot()
        self.last_context = context
        pcc_error = None if scene["reference"] is None else float(np.linalg.norm(scene["pcc"] - scene["reference"]))
        corrected_error = None if scene["reference"] is None or scene["corrected"] is None else float(np.linalg.norm(scene["corrected"] - scene["reference"]))
        self.cards["theta"].set_value(f"{scene['theta_deg']:.1f}°")
        self.cards["pcc"].set_value(self._format_xyz(scene["pcc"]))
        baseline = physical and model_name == "PCC"
        self.cards["corrected"].set_title("PCC baseline · no ML" if baseline else f"{model_name} tip · XYZ")
        self.cards["corrected_error"].set_title("PCC baseline point error" if baseline else "Selected point error")
        self.cards["corrected"].set_value(self._format_xyz(scene["corrected"]))
        self.cards["reference"].set_value(self._format_xyz(scene["reference"], empty="No test observation"))
        for key, value in self.selection_values.items():
            value.setText(self._format_xyz(scene[key], empty="No test observation" if key == "reference" else "Unavailable"))
        self.cards["reference"].set_title("Measured test endpoint · XYZ" if physical else "Synthetic reference · XYZ")
        self.cards["pcc_error"].set_value("No observation" if pcc_error is None else f"{pcc_error:.3f} mm")
        self.cards["corrected_error"].set_value("No observation" if corrected_error is None else f"{corrected_error:.3f} mm")
        if baseline:
            self.cards["improvement"].set_value("Baseline · no correction")
            self.cards["improvement"].value.setStyleSheet(f"color: {COLORS['muted']};")
        elif corrected_error is not None and pcc_error is not None and pcc_error > 1e-9:
            change = 100 * (1 - corrected_error / pcc_error)
            self.cards["improvement"].set_value(f"{change:+.1f}%")
            self.cards["improvement"].value.setStyleSheet(
                f"color: {COLORS['positive'] if change >= 0 else COLORS['danger']};")
        else:
            self.cards["improvement"].set_value("—")
            self.cards["improvement"].value.setStyleSheet(f"color: {COLORS['muted']};")
        self.cards["source"].set_value("Physical · v2 test" if physical else "Synthetic · deterministic")
        self.statusBar().showMessage(scene["reference_label"] + " · " + scene["frame_label"] +
                                     (" · " + scene["warning"] if scene["warning"] else ""))

    def focus_robot(self):
        self.view3d.focus_robot()
        self._projection_frame = "robot"
        self.last_context = None
        self.schedule_refresh()

    def reset_camera(self):
        self.focus_robot()
        self.view3d.camera_preset("Isometric")

    def fit_workspace(self):
        self.view3d.fit_workspace()
        self._projection_frame = "workspace"
        self.last_context = None
        self.schedule_refresh()

    def fit_view(self):
        self.focus_robot()

    def capture_screenshot(self):
        self._capture(viewport_only=False)

    def capture_viewport(self):
        self._capture(viewport_only=True)

    def _capture(self, *, viewport_only: bool):
        destination = ROOT / "figures" / "gui"
        destination.mkdir(parents=True, exist_ok=True)
        default_name = "robot_viewport.png" if viewport_only else "dashboard_overview.png"
        path, _ = QFileDialog.getSaveFileName(self, "Capture actual dashboard", str(destination / default_name),
                                              "PNG image (*.png)")
        if path:
            # Let the native file dialog disappear before grabbing the widgets.
            QTimer.singleShot(200, lambda: self._save_capture(path, viewport_only))

    def _save_capture(self, path: str, viewport_only: bool):
        try:
            if viewport_only:
                if not self.view3d.isVisible():
                    raise RuntimeError("Open the Robot, Compensation or Measurements workspace first")
                gl_image = self.view3d.grabFramebuffer()
                if gl_image.isNull():
                    raise RuntimeError("OpenGL framebuffer is empty")
                saved = gl_image.save(path, "PNG")
            else:
                image = self.grab()
                if self.view3d.isVisible():
                    gl_image = self.view3d.grabFramebuffer()
                    if gl_image.isNull():
                        raise RuntimeError("OpenGL framebuffer is empty")
                    painter = QPainter(image)
                    rect = QRect(self.view3d.mapTo(self, QPoint(0, 0)), self.view3d.size())
                    painter.drawImage(rect, gl_image)
                    painter.end()
                saved = image.save(path, "PNG")
            if not saved:
                raise OSError("PNG encoder could not write the file")
            self.statusBar().showMessage(f"Screenshot saved: {path}", 7000)
        except Exception as exc:
            self.statusBar().showMessage(f"Screenshot failed: {exc}", 10000)

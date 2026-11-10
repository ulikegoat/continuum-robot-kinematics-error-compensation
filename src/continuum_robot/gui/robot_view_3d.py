"""Interactive OpenGL viewport for analytical geometry and measured endpoints."""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtGui import QVector3D

from continuum_robot.gui.theme import COLORS

MODEL_COLOR = {"PCC": (.91, .54, .31, 1.), "KNN": (.72, .59, .87, 1.),
               "FFNN": (.36, .78, .55, 1.), "Ridge3": (.83, .66, .34, 1.),
               "Ridge2": (.82, .45, .53, 1.), "Linear": (.76, .78, .74, 1.)}


def _tube(curve: np.ndarray, radius: float = 1.45, sides: int = 12):
    """Illustrative constant-radius tube around the PCC centreline.

    Radius is a display convention; it is not measured robot geometry.
    """
    tangent = np.gradient(curve, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    reference = np.repeat([[0., 0., 1.]], len(curve), axis=0)
    reference[np.abs(tangent[:, 2]) > .9] = [0., 1., 0.]
    normal = np.cross(tangent, reference)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    binormal = np.cross(tangent, normal)
    angle = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    vertices = (curve[:, None, :] + radius * (np.cos(angle)[None, :, None] * normal[:, None, :]
                                              + np.sin(angle)[None, :, None] * binormal[:, None, :]))
    faces = []
    for i in range(len(curve) - 1):
        for j in range(sides):
            a = i * sides + j
            b = i * sides + (j + 1) % sides
            faces.extend(((a, b, a + sides), (b, b + sides, a + sides)))
    return vertices.reshape(-1, 3), np.asarray(faces, dtype=np.uint32)


class RobotView(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundColor(COLORS["viewport"])
        self.setMinimumSize(320, 260)
        self.opts["distance"] = 340
        self.opts["elevation"] = 22
        self.opts["azimuth"] = -75
        self._robot_center = np.array([0., 0., 50.])
        self._robot_extent = 110.
        self._workspace_center = self._robot_center.copy()
        self._workspace_extent = 110.
        self._framing = "robot"
        self._scene_initialized = False
        self._cloud_source = None
        self._cloud_style = None

        self.grid = gl.GLGridItem(color=(.39, .40, .39, .25))
        self.grid.setSize(180, 180)
        self.grid.setSpacing(20, 20)
        self.addItem(self.grid)
        self.axes = []
        for color in ((.89, .40, .37, .85), (.41, .76, .47, .85), (.47, .55, .84, .85)):
            line = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=color, width=2, antialias=True)
            self.addItem(line)
            self.axes.append(line)
        # Draw the cloud before the backbone and endpoint markers so the
        # selected configuration remains legible in dense workspace views.
        self.cloud = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=2,
                                          color=(.63, .61, .57, .15), pxMode=True)
        self.addItem(self.cloud)
        self.cloud.setVisible(False)
        self.backbone = gl.GLMeshItem(vertexes=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                                      faces=np.array([[0, 1, 2]]),
                                      color=(.91, .54, .31, 1.), smooth=True, shader="shaded")
        self.backbone.setGLOptions("opaque")
        self.addItem(self.backbone)
        self.reference_curve = gl.GLLinePlotItem(pos=np.zeros((2, 3)),
                                                 color=(.82, .72, .48, .75), width=3, antialias=True)
        self.addItem(self.reference_curve)
        self.reference_halo = self._point((.95, .81, .48, .19), 35)
        self.base_marker = self._point((.86, .86, .81, .92), 11)
        self.point_pcc = self._point((.97, .58, .34, 1.), 19)
        self.point_reference = self._point((.95, .81, .48, 1.), 20)
        self.point_corrected = self._point((.36, .78, .55, 1.), 22)
        self.error_pcc = self._line((.97, .58, .34, .85))
        self.error_corrected = self._line((.36, .78, .55, .9))
        self.correction_vector = self._line((.80, .76, .69, .85))

    def _point(self, color, size):
        item = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=color, size=size, pxMode=True)
        self.addItem(item)
        item.setVisible(False)
        return item

    def _line(self, color):
        item = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=color, width=2, antialias=True)
        self.addItem(item)
        item.setVisible(False)
        return item

    def clear_scene(self):
        for item in (self.grid, *self.axes, self.cloud, self.backbone, self.reference_curve,
                     self.reference_halo, self.base_marker, self.point_pcc,
                     self.point_reference, self.point_corrected, self.error_pcc,
                     self.error_corrected, self.correction_vector):
            item.setVisible(False)

    def set_scene(self, scene: dict, *, cloud=None, cloud_mode="Off", show_vectors=True, show_robot=True,
                  error_scale_mm: float | None = None):
        self.grid.setVisible(True)
        for axis in self.axes:
            axis.setVisible(True)
        curve = scene["pcc_curve"]
        vertices, faces = _tube(curve)
        self.backbone.setMeshData(vertexes=vertices, faces=faces)
        self.backbone.setVisible(show_robot)
        self.base_marker.setData(pos=curve[:1])
        self.base_marker.setVisible(show_robot)
        self.reference_curve.setVisible(scene["reference_curve"] is not None)
        if scene["reference_curve"] is not None:
            self.reference_curve.setData(pos=scene["reference_curve"])
        corrected_color = MODEL_COLOR.get(scene.get("display_model", "FFNN"), MODEL_COLOR["FFNN"])
        self.point_corrected.setData(color=corrected_color)
        self.error_corrected.setData(color=corrected_color)
        redundant_corrected = scene["corrected"] is not None and np.allclose(scene["corrected"], scene["pcc"])
        for item, point in ((self.point_pcc, scene["pcc"]),
                            (self.point_reference, scene["reference"]),
                            (self.point_corrected, None if redundant_corrected else scene["corrected"])):
            item.setVisible(point is not None)
            if point is not None:
                item.setData(pos=np.asarray(point, float).reshape(1, 3))
        reference = scene["reference"]
        corrected = scene["corrected"]
        self.reference_halo.setVisible(reference is not None)
        if reference is not None:
            self.reference_halo.setData(pos=np.asarray(reference, float).reshape(1, 3))
        self.error_pcc.setVisible(show_vectors and reference is not None)
        self.error_corrected.setVisible(show_vectors and reference is not None and corrected is not None and not redundant_corrected)
        self.correction_vector.setVisible(show_vectors and corrected is not None and not redundant_corrected)
        if reference is not None:
            self.error_pcc.setData(pos=np.vstack((scene["pcc"], reference)))
            if corrected is not None:
                self.error_corrected.setData(pos=np.vstack((corrected, reference)))
        if corrected is not None and not redundant_corrected:
            self.correction_vector.setData(pos=np.vstack((scene["pcc"], corrected)))
        if cloud is not None and cloud_mode != "Off":
            points, errors = cloud
            style = (cloud_mode, error_scale_mm)
            if len(points) and (cloud is not self._cloud_source or style != self._cloud_style):
                ceiling = max(float(error_scale_mm or np.percentile(errors, 95)), 1e-6)
                fraction = np.clip(errors / ceiling, 0, 1)
                # Neutral gray -> copper, with opacity low enough to keep
                # the selected backbone and endpoints visually dominant.
                opacity = .17 if cloud_mode == "Light" else .24
                rgba = np.column_stack((.58 + .33 * fraction, .58 - .23 * fraction,
                                        .56 - .30 * fraction, np.full(len(points), opacity)))
                self.cloud.setData(pos=points, color=rgba,
                                   size=2 if cloud_mode == "Light" else 2.5, pxMode=True)
                self._cloud_source = cloud
                self._cloud_style = style
            self.cloud.setVisible(bool(len(points)))
        else:
            self.cloud.setVisible(False)
        robot_points = [curve]
        if reference is not None:
            robot_points.append(np.asarray(reference).reshape(1, 3))
        if corrected is not None:
            robot_points.append(np.asarray(corrected).reshape(1, 3))
        robot = np.vstack(robot_points)
        workspace = np.vstack((robot, cloud[0])) if cloud is not None and len(cloud[0]) else robot
        self._robot_center = (robot.min(0) + robot.max(0)) / 2
        self._robot_extent = max(float(np.ptp(robot, axis=0).max()), 55.)
        self._workspace_center = (workspace.min(0) + workspace.max(0)) / 2
        self._workspace_extent = max(float(np.ptp(workspace, axis=0).max()), 55.)
        self._place_frame(curve)
        if not self._scene_initialized:
            self.focus_robot()
            self._scene_initialized = True

    def _place_frame(self, curve: np.ndarray):
        base = curve[0]
        floor = float(base[2] - 5)
        self.grid.resetTransform()
        self.grid.translate(float(base[0]), float(base[1]), floor)
        origin = np.array([base[0] - 25, base[1] - 25, floor])
        for i, item in enumerate(self.axes):
            endpoint = origin.copy()
            endpoint[i] += 36
            item.setData(pos=np.vstack((origin, endpoint)))

    def focus_robot(self):
        self._framing = "robot"
        self.setCameraPosition(pos=QVector3D(*map(float, self._robot_center)),
                               distance=max(180., self._robot_extent * 2.9))

    def fit_workspace(self):
        self._framing = "workspace"
        self.setCameraPosition(pos=QVector3D(*map(float, self._workspace_center)),
                               distance=max(220., self._workspace_extent * 2.9))

    def fit_view(self):
        """Compatibility alias for the former Fit view control."""
        self.focus_robot()

    def camera_preset(self, name: str):
        angles = {"Isometric": (22, -75), "Top": (90, -90),
                  "Front": (0, -90), "Side": (0, 0)}
        elevation, azimuth = angles[name]
        center = self._robot_center if self._framing == "robot" else self._workspace_center
        extent = self._robot_extent if self._framing == "robot" else self._workspace_extent
        self.setCameraPosition(pos=QVector3D(*map(float, center)),
                               elevation=elevation, azimuth=azimuth,
                               distance=max(180., extent * 2.9))

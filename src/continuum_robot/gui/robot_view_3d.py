"""Interactive OpenGL viewport for analytical geometry and measured endpoints."""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtGui import QVector3D

MODEL_COLOR = {"PCC": (.96, .65, .39, 1.), "KNN": (.63, .55, 1., 1.),
               "FFNN": (.31, .83, .72, 1.), "Ridge3": (.83, .66, .44, 1.),
               "Ridge2": (.72, .60, .85, 1.), "Linear": (.50, .68, .83, 1.)}


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
        self.setBackgroundColor("#0A1420")
        self.setMinimumSize(380, 340)
        self.opts["distance"] = 340
        self.opts["elevation"] = 22
        self.opts["azimuth"] = -75
        self._center = np.array([0., 0., 50.])
        self._extent = 130.
        self._scene_initialized = False
        self._show_cloud = True
        self._show_vectors = True
        self._cloud_source = None

        self.grid = gl.GLGridItem(color=(.21, .35, .45, .36))
        self.grid.setSize(180, 180)
        self.grid.setSpacing(20, 20)
        self.addItem(self.grid)
        self.axes = []
        for color in ((1., .44, .38, .85), (.39, .87, .53, .85), (.42, .68, 1., .9)):
            line = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=color, width=2, antialias=True)
            self.addItem(line)
            self.axes.append(line)
        self.backbone = gl.GLMeshItem(vertexes=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                                      faces=np.array([[0, 1, 2]]),
                                      color=(.96, .63, .39, 1.), smooth=True, shader="shaded")
        self.backbone.setGLOptions("opaque")
        self.addItem(self.backbone)
        self.reference_curve = gl.GLLinePlotItem(pos=np.zeros((2, 3)),
                                                 color=(.66, .74, .88, .75), width=3, antialias=True)
        self.addItem(self.reference_curve)
        self.point_pcc = self._point((1., .65, .39, 1.), 13)
        self.point_reference = self._point((.75, .84, .98, 1.), 15)
        self.point_corrected = self._point((.31, .83, .72, 1.), 17)
        self.error_pcc = self._line((1., .58, .41, .85))
        self.error_corrected = self._line((.31, .83, .72, .9))
        self.cloud = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=3,
                                          color=(.36, .66, .88, .4), pxMode=True)
        self.addItem(self.cloud)
        self.cloud.setVisible(False)

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

    def set_scene(self, scene: dict, *, cloud=None, show_cloud=True, show_vectors=True):
        curve = scene["pcc_curve"]
        vertices, faces = _tube(curve)
        self.backbone.setMeshData(vertexes=vertices, faces=faces)
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
        self.error_pcc.setVisible(show_vectors and reference is not None)
        self.error_corrected.setVisible(show_vectors and reference is not None and corrected is not None and not redundant_corrected)
        if reference is not None:
            self.error_pcc.setData(pos=np.vstack((scene["pcc"], reference)))
            if corrected is not None:
                self.error_corrected.setData(pos=np.vstack((corrected, reference)))
        self._show_cloud = show_cloud
        self._show_vectors = show_vectors
        if cloud is not None and show_cloud:
            points, errors = cloud
            if len(points) and cloud is not self._cloud_source:
                fraction = np.clip(errors / max(float(np.percentile(errors, 95)), 1e-6), 0, 1)
                rgba = np.column_stack((.22 + .72 * fraction, .70 - .40 * fraction,
                                        .85 - .54 * fraction, np.full(len(points), .52)))
                self.cloud.setData(pos=points, color=rgba, size=3, pxMode=True)
                self._cloud_source = cloud
            self.cloud.setVisible(bool(len(points)))
        else:
            self.cloud.setVisible(False)
        all_points = [curve]
        if reference is not None:
            all_points.append(np.asarray(reference).reshape(1, 3))
        if cloud is not None and show_cloud and len(cloud[0]):
            all_points.append(cloud[0])
        stacked = np.vstack(all_points)
        self._center = (stacked.min(0) + stacked.max(0)) / 2
        self._extent = max(float(np.ptp(stacked, axis=0).max()), 65.)
        self._place_frame(stacked)
        if not self._scene_initialized:
            self.fit_view()
            self._scene_initialized = True

    def _place_frame(self, points: np.ndarray):
        floor = float(points[:, 2].min() - 8)
        self.grid.resetTransform()
        self.grid.translate(float(self._center[0]), float(self._center[1]), floor)
        origin = np.array([self._center[0] - 45, self._center[1] - 45, floor])
        for i, item in enumerate(self.axes):
            endpoint = origin.copy()
            endpoint[i] += 36
            item.setData(pos=np.vstack((origin, endpoint)))

    def fit_view(self):
        self.setCameraPosition(pos=QVector3D(*map(float, self._center)),
                               distance=max(230., self._extent * 3.2))

    def camera_preset(self, name: str):
        angles = {"Isometric": (22, -75), "Top": (90, -90),
                  "Front": (0, -90), "Side": (0, 0)}
        elevation, azimuth = angles[name]
        self.setCameraPosition(pos=QVector3D(*map(float, self._center)),
                               elevation=elevation, azimuth=azimuth,
                               distance=max(230., self._extent * 3.2))

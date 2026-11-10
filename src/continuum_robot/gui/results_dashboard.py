"""Interactive plots derived only from saved Phase 6 v2 CSV files."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtWidgets import QHBoxLayout, QLabel, QTabWidget, QVBoxLayout, QWidget

from continuum_robot.gui.model_adapter import PhysicalAdapter, V2_RESULTS
from continuum_robot.gui.theme import COLORS

MODEL_COLORS = {name: COLORS[key] for name, key in (
    ("PCC", "pcc"), ("Linear", "linear"), ("Ridge2", "ridge2"),
    ("Ridge3", "ridge3"), ("KNN", "knn"), ("FFNN", "ffnn"))}
MODELS = ("PCC", "Linear", "Ridge2", "Ridge3", "KNN", "FFNN")


def _plot(title: str, left: str, bottom: str) -> pg.PlotWidget:
    plot = pg.PlotWidget(title=title)
    plot.setBackground(COLORS["surface"])
    plot.showGrid(x=True, y=True, alpha=.13)
    plot.setLabel("left", left)
    plot.setLabel("bottom", bottom)
    plot.getAxis("left").setTextPen(COLORS["muted"])
    plot.getAxis("bottom").setTextPen(COLORS["muted"])
    plot.getAxis("left").setPen(COLORS["border"])
    plot.getAxis("bottom").setPen(COLORS["border"])
    plot.getPlotItem().getViewBox().setMenuEnabled(False)
    plot.setMinimumHeight(170)
    return plot


class ResultsDashboard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        self.note = QLabel("Select a Phase 6 v2 experiment to inspect held-out physical measurements.")
        self.note.setObjectName("Muted")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.rmse = _plot("Test RMSE by model", "RMSE [mm]", "Model")
        self.cdf = _plot("Test error distribution", "Cumulative fraction", "Position error [mm]")
        self.sparse = _plot("Sparse training grid", "Test RMSE [mm]", "Grid spacing [mm]")
        self.region = _plot("Excluded workspace regions", "Test RMSE [mm]", "Region")
        self.axis = _plot("Per-axis test RMSE · PCC / KNN / FFNN", "RMSE [mm]", "Cartesian axis")
        for title, plots in (("Model accuracy", (self.rmse, self.cdf)),
                             ("Generalization", (self.sparse, self.region)),
                             ("Axis detail", (self.axis,))):
            page = QWidget()
            row = QHBoxLayout(page)
            row.setContentsMargins(5, 5, 5, 5)
            row.setSpacing(12)
            for plot in plots:
                row.addWidget(plot, 1)
            self.tabs.addTab(page, title)

    def clear(self, message: str) -> None:
        self.note.setText(message)
        for plot in (self.rmse, self.cdf, self.sparse, self.region, self.axis):
            plot.clear()
            legend = plot.getPlotItem().legend
            if legend is not None:
                legend.clear()

    def show_experiment(self, adapter: PhysicalAdapter) -> None:
        self.clear(f"{adapter.experiment} · physical TEST rows · coordinates assumed mm · train-registered PCC")
        metrics = adapter.metrics.set_index("method")
        winner = metrics.RMSE_norm.idxmin()
        self.note.setText(f"{adapter.experiment} · N={int(metrics.loc['PCC', 'N']):,} held-out physical endpoints · "
                          f"lowest RMSE: {winner} {float(metrics.loc[winner, 'RMSE_norm']):.3f} mm · "
                          "registered PCC uses training data")
        self.rmse.getAxis("bottom").setTicks([[(i, m) for i, m in enumerate(MODELS)]])
        for i, method in enumerate(MODELS):
            self.rmse.addItem(pg.BarGraphItem(x=[i], height=[float(metrics.loc[method, "RMSE_norm"])],
                                              width=.72, brush=MODEL_COLORS[method]))
        self.rmse.setXRange(-.7, len(MODELS) - .3, padding=0)
        self.cdf.addLegend(offset=(8, 8))
        for method in ("PCC", "KNN", "FFNN"):
            values = np.sort(adapter.predictions[f"e_{method}"].to_numpy(float))
            indices = np.linspace(0, len(values) - 1, min(len(values), 3000), dtype=int)
            self.cdf.plot(values[indices], (indices + 1) / len(values),
                          pen=pg.mkPen(MODEL_COLORS[method], width=2), name=method)
        self.axis.getAxis("bottom").setTicks([[(i, axis) for i, axis in enumerate(("X", "Y", "Z"))]])
        self.axis.addLegend(offset=(8, 8))
        for index, method in enumerate(("PCC", "KNN", "FFNN")):
            self.axis.plot([], [], pen=pg.mkPen(MODEL_COLORS[method], width=4), name=method)
            offsets = np.arange(3, dtype=float) + (index - 1) * .24
            values = [float(metrics.loc[method, f"RMSE_{axis}"]) for axis in "XYZ"]
            self.axis.addItem(pg.BarGraphItem(x=offsets, height=values, width=.22,
                                              brush=MODEL_COLORS[method]))
        self.axis.setXRange(-.5, 2.5, padding=0)
        all_path = V2_RESULTS / "all_metrics.csv"
        if not all_path.exists():
            self.note.setText(self.note.text() + " · aggregate plots unavailable: all_metrics.csv missing")
            return
        all_metrics = pd.read_csv(all_path)
        self.sparse.addLegend(offset=(8, 8))
        self.region.addLegend(offset=(8, 8))
        for method in ("PCC", "KNN", "FFNN"):
            selected = all_metrics[all_metrics.method == method]
            sparse = selected[selected.experiment.str.startswith("sparse_")].copy()
            sparse["spacing"] = sparse.experiment.str.split("_").str[-1].astype(float) / 10
            sparse = sparse.sort_values("spacing")
            self.sparse.plot(sparse.spacing.to_numpy(float), sparse.RMSE_norm.to_numpy(float),
                             pen=pg.mkPen(MODEL_COLORS[method], width=2), symbol="o",
                             symbolBrush=MODEL_COLORS[method], name=method)
            region = selected[selected.experiment.str.startswith("region_")].copy()
            region["number"] = region.experiment.str.split("_").str[-1].astype(int)
            region = region.sort_values("number")
            self.region.plot(region.number.to_numpy(float), region.RMSE_norm.to_numpy(float),
                             pen=pg.mkPen(MODEL_COLORS[method], width=2), symbol="o",
                             symbolBrush=MODEL_COLORS[method], name=method)
        self.region.getAxis("bottom").setTicks([[(1, "Plane 1"), (2, "Plane 2"), (3, "Plane 3")]])

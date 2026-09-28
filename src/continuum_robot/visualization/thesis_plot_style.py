"""Consistent, print-friendly Matplotlib styling for synthetic validation figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "PCC": "#B45F45",
    "PCC + Linear Regression": "#667C9A",
    "PCC + Polynomial Ridge": "#85698D",
    "PCC + KNN": "#A1894D",
    "PCC + NN": "#187C74",
    "PCC IK": "#B45F45",
    "PCC+NN IK": "#187C74",
    "Synthetic reference": "#38444C",
    "Target": "#38444C",
}


def apply_thesis_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.edgecolor": "#718088",
        "axes.labelcolor": "#26333A",
        "text.color": "#26333A",
        "xtick.color": "#38444C",
        "ytick.color": "#38444C",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def light_grid(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, color="#D9E1E4", linewidth=0.7)
    ax.set_axisbelow(True)


def equal_3d_axes(ax, points: np.ndarray) -> None:
    """Give X, Y, and Z the same physical scale without altering point values."""
    bounds = np.asarray(points, dtype=float)
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    center = (low + high) / 2
    half_span = max(float(np.max(high - low)) * 0.56, 1.0)
    for setter, mid in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(mid - half_span, mid + half_span)
    ax.set_box_aspect((1, 1, 1))


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.16,
                facecolor="white")
    plt.close(fig)

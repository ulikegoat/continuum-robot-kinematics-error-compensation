"""Render the synthetic compensation workflow as a thesis figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from continuum_robot.common.paths import FIGURES
from continuum_robot.visualization.thesis_plot_style import apply_thesis_style, save_figure


OUTPUT = FIGURES / "methodology" / "pipeline_synthetic_compensation.png"


def box(ax, x, y, w, h, label, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                edgecolor="#6B7C83", facecolor=fill, linewidth=1.1))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=9.5, color="#26333A", linespacing=1.25)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.3, color="#718088"))


def route(ax, points):
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color="#718088", linewidth=1.3)
    arrow(ax, points[-2], points[-1])


def main():
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    blue, amber, green = "#E5EEF2", "#F5EBE2", "#E5F0EC"
    box(ax, .03, .47, .15, .12, "Tendon input\n$\\Delta l_1, \\Delta l_2, \\Delta l_3$", blue)
    box(ax, .24, .75, .16, .12, "Ideal PCC model\n$f_{\\mathrm{PCC}}$", blue)
    box(ax, .24, .45, .16, .12, "Synthetic reference\nmodel $f_{\\mathrm{syn}}$", amber)
    box(ax, .46, .75, .16, .12, "PCC position\n$p_{\\mathrm{PCC}}$", blue)
    box(ax, .46, .45, .16, .12, "Reference position\n$p_{\\mathrm{syn}}$", amber)
    box(ax, .69, .58, .19, .12, "Residual error\n$\\Delta p=p_{\\mathrm{syn}}-p_{\\mathrm{PCC}}$", green)
    box(ax, .69, .78, .19, .10, "NN training\n$(\\Delta l,\\Delta p)$", green)
    box(ax, .69, .36, .19, .10, "Trained NN\n$f_{\\mathrm{NN}}(\\Delta l)$", green)
    box(ax, .41, .14, .27, .13, "Compensated forward model\n$f_{\\mathrm{PCC+NN}}=f_{\\mathrm{PCC}}+f_{\\mathrm{NN}}$", green)
    box(ax, .05, .14, .24, .13, "Before compensation\n$p_{\\mathrm{PCC}}-p_{\\mathrm{syn}}$", amber)
    box(ax, .75, .14, .21, .13, "After compensation\n$p_{\\mathrm{PCC+NN}}-p_{\\mathrm{syn}}$", green)

    for start, end in [
        ((.18, .56), (.24, .81)), ((.18, .50), (.24, .51)),
        ((.40, .81), (.46, .81)), ((.40, .51), (.46, .51)),
        ((.62, .81), (.69, .67)), ((.62, .51), (.69, .62)),
        ((.785, .70), (.785, .78)), ((.69, .40), (.68, .25)),
        ((.68, .20), (.75, .20)),
    ]:
        arrow(ax, start, end)
    route(ax, [(.88, .83), (.92, .83), (.92, .41), (.88, .41)])
    route(ax, [(.18, .59), (.18, .92), (.785, .92), (.785, .88)])
    route(ax, [(.46, .81), (.43, .81), (.43, .31), (.53, .31), (.53, .27)])
    route(ax, [(.43, .31), (.17, .31), (.17, .27)])
    ax.text(.5, .96, "Synthetic forward-model error compensation", ha="center",
            va="center", fontsize=15, weight="semibold", color="#26333A")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, OUTPUT)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()

"""Render the authorization workflow and paired benchmark outcomes for the paper."""

import argparse
import sys
from math import ceil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MultipleLocator, NullLocator

from effect_display import COLORS, scores
from effect_results import ARMS, EXPANSION_DEST, ROOT
from effect_results import DEST as INITIAL_RESULTS

DEST = ROOT / "paper" / "generated"
INK = "#20303a"
MUTED = "#536470"
GREEN = "#117563"
RUST = "#9a402c"
LINE = "#d5dfe4"


def workflow():
    fig, ax = plt.subplots(figsize=(6.4, 2.29))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def box(x, y, width, height, title, body, fill, stroke):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.008,rounding_size=0.014",
                facecolor=fill,
                edgecolor=stroke,
                linewidth=0.8,
            )
        )
        ax.text(
            x + 0.018,
            y + height - 0.056,
            title,
            weight="bold",
            fontsize=8,
            va="top",
            color=INK,
        )
        if body:
            ax.text(
                x + 0.018,
                y + height - 0.116,
                body,
                fontsize=7.1,
                linespacing=1.35,
                va="top",
                color=MUTED,
            )

    def arrow(points, color=MUTED):
        if len(points) > 2:
            ax.plot(*zip(*points[:-1], strict=True), color=color, linewidth=0.85)
        ax.add_patch(
            FancyArrowPatch(
                *points[-2:],
                arrowstyle="-|>",
                mutation_scale=8,
                color=color,
                linewidth=0.85,
                shrinkA=0,
                shrinkB=0,
            )
        )

    ax.text(0.020, 0.953, "Inputs", color=MUTED, weight="bold", fontsize=7.6)
    box(
        0.020,
        0.58,
        0.225,
        0.27,
        "User-confirmed grant",
        "Allowed objects and targets\nLimits and conditions",
        "#f5f8fa",
        LINE,
    )
    box(
        0.020,
        0.19,
        0.225,
        0.27,
        "Tool effect preview",
        "Actual objects and targets\nQuantity per effect",
        "#f5f8fa",
        LINE,
    )
    box(
        0.322,
        0.345,
        0.225,
        0.35,
        "Policy as code",
        "Check operation and scope.\nSum quantities by object.\nApply both limits.",
        "#eef7f3",
        "#9ebfb7",
    )
    box(
        0.633,
        0.391,
        0.178,
        0.25,
        "Model review",
        "Scope and condition\nfrom current records",
        "#f5f3fa",
        "#c5b9d6",
    )
    box(0.877, 0.71, 0.100, 0.125, "Allow", "", "#eef7f3", "#9ebfb7")
    box(0.873, 0.185, 0.105, 0.125, "Hold", "", "#fbf1ed", "#d7ada1")

    arrow([(0.253, 0.71), (0.285, 0.71), (0.285, 0.598), (0.312, 0.598)])
    arrow([(0.253, 0.325), (0.285, 0.325), (0.285, 0.435), (0.312, 0.435)])
    arrow([(0.444, 0.704), (0.444, 0.772), (0.867, 0.772)], GREEN)
    arrow([(0.444, 0.336), (0.444, 0.247), (0.863, 0.247)], RUST)
    arrow([(0.557, 0.517), (0.623, 0.517)])
    arrow([(0.821, 0.553), (0.852, 0.553), (0.852, 0.72), (0.870, 0.72)], GREEN)
    arrow([(0.821, 0.461), (0.848, 0.461), (0.848, 0.300), (0.864, 0.300)], RUST)
    ax.text(0.548, 0.807, "Checks pass, no condition", color=GREEN, fontsize=7.0)
    ax.text(0.485, 0.201, "A check fails", color=RUST, fontsize=7.0)
    ax.text(
        0.588,
        0.571,
        "Condition\nremains",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=6.3,
    )
    ax.text(0.640, 0.661, "Both answers affirmative", color=GREEN, fontsize=6.7)
    ax.text(0.662, 0.346, "Otherwise withhold", color=RUST, fontsize=6.7)
    return fig


def outcomes(*, result_dir=INITIAL_RESULTS):
    table = scores(dest=result_dir)
    names = tuple(ARMS.values())
    colors = [COLORS[arm] for arm in ARMS]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.20), sharey=True)
    fig.subplots_adjust(left=0.204, right=0.978, top=0.84, bottom=0.257, wspace=0.28)
    for ax, panel in zip(
        axes,
        (
            ("a", "Legal releases", "legal_allow", "legal_n"),
            ("b", "False releases", "false_allow", "unsafe_n"),
        ),
        strict=True,
    ):
        letter, title, column, denominator = panel
        rows = [table["all", arm] for arm in ARMS]
        values = [int(r[column]) for r in rows]
        top = int(rows[0][denominator]) if letter == "a" else max(1, *values)
        tick = (10 if top < 35 else 40) if top < 180 else (100 if top < 500 else 200)
        limit = max(ceil(1.18 * top), tick)
        ax.set_xlim(0, limit)
        ax.set_ylim(len(names) - 0.45, -0.55)
        ax.barh(range(len(names)), values, height=0.46, color=colors, zorder=2)
        ax.set_yticks(range(len(names)), names)
        ax.tick_params(axis="y", length=0, pad=7)
        ax.tick_params(axis="x", colors=MUTED, length=3)
        ax.xaxis.set_major_locator(MultipleLocator(tick))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_axisbelow(True)
        ax.spines[["left", "top", "right"]].set_visible(False)
        ax.spines["bottom"].set_color(LINE)
        ax.set_xlabel(f"Of {rows[0][denominator]} requests", color=MUTED, labelpad=5)
        ax.set_title(title, loc="left", fontweight="bold", pad=7)
        ax.text(
            -0.054,
            1.105,
            letter,
            transform=ax.transAxes,
            weight="bold",
            ha="right",
            va="center",
            fontsize=9,
            gid=f"panel-label-{letter}",
        )
        for y, value in enumerate(values):
            ax.annotate(
                str(value),
                (value, y),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                color=INK,
                weight="bold",
                fontsize=7.4,
            )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-scripts",
        type=Path,
        help="Directory containing the optional figure auditor",
    )
    parser.add_argument(
        "--qa-dir", type=Path, help="Save layout reports and preview images here"
    )
    args = parser.parse_args()
    if args.audit_scripts:
        sys.path.insert(0, str(args.audit_scripts.resolve()))
        from audit_panel_alignment import require_matplotlib_panel_alignment
    if args.qa_dir:
        args.qa_dir.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
        }
    )
    figures = [("effect_workflow", workflow), ("effect_outcomes", outcomes)]
    if (EXPANSION_DEST / "scores.csv").is_file():
        figures.append(
            ("effect_expansion_outcomes", lambda: outcomes(result_dir=EXPANSION_DEST))
        )
    for name, draw in figures:
        fig = draw()
        fig.canvas.draw()
        if args.audit_scripts:
            require_matplotlib_panel_alignment(
                fig,
                strict=True,
                json_out=args.qa_dir / f"{name}.alignment.json"
                if args.qa_dir
                else None,
            )
        fig.savefig(DEST / f"{name}.pdf")
        if args.qa_dir:
            fig.savefig(args.qa_dir / f"{name}.svg")
            fig.savefig(
                args.qa_dir / f"{name}.tiff",
                dpi=600,
                pil_kwargs={"compression": "tiff_lzw"},
            )
            fig.savefig(args.qa_dir / f"{name}.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()

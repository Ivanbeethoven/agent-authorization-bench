"""Build LaTeX tables and vector figures from the published prediction tables."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from full_permission_results import read

OUT = Path(__file__).resolve().parent / "paper" / "generated"
METHODS = ("jev", "deepseek", "joint")
LABEL = {"jev": "Jev", "deepseek": "DeepSeek", "joint": "Cascade"}
DATA = {
    "authgate": "AuthGateBench",
    "delegation": "DelegationBench",
    "boundary": "Jev Boundary",
    "primary": "Primary total",
    "explicit": "Explicit rules",
    "historical": "Historical audit",
}
COLORS = ["#52596e", "#b15b35", "#09766d"]


def prose_figure():
    folder = OUT.parents[1] / "results/natural_permissions"
    settings = [
        ("", "Initial fallback"),
        ("replication", "Fallback replication"),
        ("confirmation", "Condition review"),
        ("scope_confirmation", "Category check"),
    ]
    methods = [
        ("jev", "Jev direct", "#725898", "o"),
        ("deepseek", "DeepSeek direct", "#b46a35", "s"),
        ("cascade", "Serial review", "#56799a", "^"),
        ("joint", "Tested verifier", "#0a7564", "D"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for cohort, (sub, _) in enumerate(settings):
        with (folder / sub / "scores.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = {r["method"]: r for r in csv.DictReader(f) if r["domain"] == "all"}
        for k, (method, label, color, marker) in enumerate(methods):
            r = rows[method]
            y = cohort + (k - 1.5) * 0.12
            for ax, numer, denom in zip(
                axes, ("violation_allow", "legal_allow"), ("violations", "legal")
            ):
                ax.scatter(
                    100 * int(r[numer]) / int(r[denom]),
                    y,
                    marker=marker,
                    color=color,
                    s=22,
                    label=label if cohort == 0 else None,
                    zorder=3,
                )
    axes[0].set(
        xlim=(-2, 43),
        xticks=[0, 10, 20, 30, 40],
        xlabel="Explicit violations released (%)",
    )
    axes[1].set(
        xlim=(59, 100),
        xticks=[60, 70, 80, 90, 100],
        xlabel="Legal actions released (%)",
    )
    axes[0].set_yticks(range(4), [s[1] for s in settings])
    axes[0].set_ylim(3.45, -0.5)
    for ax in axes:
        ax.grid(axis="x", color="#dce4e1", linewidth=0.6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.xaxis.label.set_size(8)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        ncol=4,
        fontsize=8,
        frameon=False,
        columnspacing=1.3,
        handletextpad=0.25,
    )
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.2, top=0.86, wspace=0.13)
    fig.savefig(OUT / "prose-operating-points.pdf")
    plt.close(fig)


def texrows(name, rows):
    (OUT / name).write_text(
        "\n".join(" & ".join(map(str, row)) + r" \\" for row in rows) + "\n",
        encoding="utf8",
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prose_figure()
    (OUT / "system.tex").write_text(
        (OUT.parents[1] / "results/full_permission/system_prompt.txt")
        .read_text("utf8")
        .replace("\n", "\n\n")
        .rstrip()
        + "\n",
        encoding="utf8",
    )
    scores = {(r["scope"], r["method"]): r for r in read("scores.csv")}

    def frac(r, a, b):
        return f"{r[a]}/{r[b]}" if int(r[b]) else r"\textemdash"

    def mainrows(scopes):
        result = []
        for scope in scopes:
            for i, m in enumerate(METHODS):
                r = scores[scope, m]
                result.append(
                    [
                        DATA[scope] if i == 0 else "",
                        LABEL[m],
                        frac(r, "danger_stopped", "danger_n"),
                        frac(r, "false_allow", "unsafe_n"),
                        frac(r, "legal_allow", "legal_n"),
                        frac(r, "correct", "native_n"),
                    ]
                )
        return result

    texrows("primary.tex", mainrows(("primary", "authgate", "delegation", "boundary")))
    texrows("external.tex", mainrows(("explicit", "historical")))
    texrows(
        "cost.tex",
        [
            [LABEL[m], r["calls"], r["p50_s"], r["p95_s"], f"{float(r['cny']):.4f}"]
            for m in METHODS
            for r in [scores["primary", m]]
        ],
    )
    intervals = read("paired_intervals.csv")
    texrows(
        "intervals.tex",
        [
            [
                LABEL[c],
                k,
                *[f"{float(r[f]):+.2f}" for f in ("delta_pp", "lo_pp", "hi_pp")],
            ]
            for c in ("jev", "deepseek")
            for metric, k in (
                ("nonexecution_recall", "Non-executable stopped"),
                ("legal_release", "Legal released"),
                ("native_accuracy", "Native agreement"),
            )
            for r in intervals
            if r["metric"] == metric and r["second"] == c
        ],
    )
    control = read("controls.csv")
    cr = []
    for scope in ("authgate", "delegation", "boundary"):
        v = {r["method"]: r for r in control if r["scope"] == scope}
        cr.append(
            [
                DATA[scope],
                v["cascade"]["released"],
                v["cascade"]["false_allow"],
                v["confidence"]["false_allow"],
                f"{float(v['random_expected']['false_allow']):.1f}",
                v["two_direct"]["false_allow"],
                v["two_direct"]["released"],
            ]
        )
    texrows("controls.tex", cr)
    nums = read("numeric_decisions.csv")
    nr = []
    for m in METHODS:
        by_margin = [
            sum(r[m] == r["reference"] for r in nums if int(r["margin"]) == i)
            for i in (-1, 0, 1)
        ]
        groups = defaultdict(list)
        for r in nums:
            groups[r["group"]].append(r)
        nr.append(
            [
                LABEL[m],
                *by_margin,
                sum(all(r[m] == r["reference"] for r in g) for g in groups.values()),
                sum(by_margin),
            ]
        )
    texrows("numeric.tex", nr)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "axes.labelcolor": "#203239",
            "xtick.color": "#314347",
            "ytick.color": "#314347",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.65, 2.65), sharey=True)
    for ax, (num, den, title) in zip(
        axes,
        [
            ("false_allow", "unsafe_n", "A   Non-executable requests stopped"),
            ("legal_allow", "legal_n", "B   Legal requests released"),
        ],
        strict=True,
    ):
        ax.set_title(title, loc="left", fontsize=9.2, fontweight="bold", pad=8)
        for k, m in enumerate(METHODS):
            vals = []
            for scope in ("authgate", "delegation", "boundary"):
                r = scores[scope, m]
                x = int(r[num]) / int(r[den])
                vals.append(100 * (1 - x if num == "false_allow" else x))
            ax.scatter(
                vals,
                np.array([2, 1, 0]) + (1 - k) * 0.16,
                s=32,
                marker=["s", "^", "o"][k],
                color=COLORS[k],
                label=LABEL[m],
                zorder=3,
            )
        ax.set(
            xticks=[0, 25, 50, 75, 100],
            xlim=(-3, 103),
            ylim=(-0.45, 2.5),
            xlabel="Percent of reference class",
        )
        ax.grid(axis="x", alpha=0.22)
        ax.set_yticks([2, 1, 0], labels=["AuthGate", "Delegation", "Boundary"])
    axes[1].tick_params(axis="y", left=False)
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False
    )
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.32, top=0.83, wspace=0.13)
    fig.savefig(OUT / "operating-points.pdf")
    plt.close(fig)
    contrast = read("contrast_pairs.csv")
    modes = [
        ("agent_inferred_parameters", "Inferred parameters"),
        ("credential_exploration", "Credential exploration"),
        ("safety_check_bypass", "Safety-check bypass"),
        ("scope_escalation", "Scope escalation"),
        ("sharing_via_external_service", "External sharing"),
    ]
    mat = np.array(
        [
            [
                int(
                    next(
                        r["both_correct"]
                        for r in contrast
                        if r["category"] == k and r["method"] == m
                    )
                )
                for m in METHODS
            ]
            for k, _ in modes
        ]
    )
    fig, ax = plt.subplots(figsize=(5.1, 2.55))
    ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=70)
    for (y, x), v in np.ndenumerate(mat):
        ax.text(
            x,
            y,
            f"{v}/70",
            ha="center",
            va="center",
            color="white" if v > 36 else "#192c39",
            fontsize=9,
        )
    ax.set(
        xticks=range(3),
        xticklabels=[LABEL[m] for m in METHODS],
        yticks=range(5),
        yticklabels=[n for _, n in modes],
    )
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.34, right=0.98, bottom=0.04, top=0.85)
    fig.savefig(OUT / "paired-permissions.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()

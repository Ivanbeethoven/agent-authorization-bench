"""Build the portable research report from the published score and usage tables."""

import argparse
import base64
import csv
import io
import json
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from shared_report import DISPLAY as METHODS
from strategy_selection import select_strategies

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "Agent权限管理研究总报告.html"
BENCH = {
    "jev": "Jev Boundary",
    "authgate": "AuthGateBench",
    "delegation": "DelegationBench",
    "authbench": "AuthBench",
    "authorization": "AuthorizationBench",
    "yolofs": "YoloFS-Ext",
    "tsbench": "TS-Bench",
    "agentdojo": "AgentDojo",
    "agentdam": "AgentDAM",
    "privacylens": "PrivacyLens",
    "privacylive": "PrivacyLens-Live",
}
MODELS = {"jev": "Jev", "deepseek": "DeepSeek", "aliyun": "百炼 Decision"}
PHASES = {
    "pilot": "11 项直接判断",
    "review": "提示与复审对照",
    "risk": "风险决策树",
    "coverage": "分支覆盖诊断",
    "conditions": "授权条件与遮蔽",
    "confirmation": "双模型明确授权",
    "parameter_calibration": "历史参数校准",
    "parameter_test": "公开数据逐参数检验",
    "parameter_versions": "合成版本配对",
    "parameter_final": "最终公开配对",
    "reply_validation": "最小干预及对照",
    "reply_precedent": "跨家族示例",
    "shared_review": "先读授权的复审",
    "commitment_review": "对外承诺配对验证",
    "retained_validation": "保留策略独立复测",
    "jev_triage": "Jev 选择审查模型",
    "parameter_router": "初判后的逐项核对",
    "parameter_router_extra": "逐项核对补充情境",
}
STAGE_GROUPS = {
    "三模型直接判断": ("pilot",),
    "提示与风险决策树": ("review", "risk", "coverage"),
    "授权条件与双模型确认": ("conditions", "confirmation"),
    "逐参数复审": (
        "parameter_calibration",
        "parameter_test",
        "parameter_versions",
        "parameter_final",
    ),
    "最小干预与专家示例": ("reply_validation", "reply_precedent"),
    "共同错误与承诺复审": ("shared_review", "commitment_review"),
    "保留策略独立复测": ("retained_validation",),
    "Jev 选择审查模型": ("jev_triage",),
    "初判后的逐项核对": ("parameter_router", "parameter_router_extra"),
}
COHORTS = {
    "new_delegation": "DelegationBench · 新请求",
    "new_boundary": "Jev Boundary · 新家族",
    "final_pairs": "AuthGateBench · 历史回归",
}
TEAL, AMBER, INK, LINE = "#176c6a", "#ba532e", "#1d333d", "#dde5e4"


def read(path):
    with (RESULTS / path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def inputs():
    ledgers = [
        "model_study/usage.csv",
        "authorization_conditions/usage.csv",
        "authorization_conditions/confirmation_usage.csv",
        "reply_review/provider_usage.csv",
        "shared_review/provider_usage.csv",
        "retained_validation/provider_usage.csv",
        "jev_triage/provider_usage.csv",
        "parameter_router/provider_usage.csv",
    ]
    ledgers.extend(
        f"parameter_review/{x}_usage.csv" for x in ("old", "fresh", "versions", "final")
    )
    usage = [r for file in ledgers for r in read(file)]
    if (
        len({(r["study"], r["provider"]) for r in usage}) != len(usage)
        or {r["study"] for r in usage} != PHASES.keys()
    ):
        raise ValueError("Study ledger incomplete or duplicated")
    history = read("model_study/pilot_scores.csv")
    if len(history) != 33 or any(int(row["samples"]) != 50 for row in history):
        raise ValueError("Initial benchmark design changed")
    runs = read("reply_review/scores.csv")
    runs.extend(
        r
        for r in read("reply_review/codex_scores.csv")
        if r["split"] in ("fresh", "regression")
    )
    extras = {
        (r["set"], r["method"]): r
        for r in read("parameter_review/final_scores.csv") + runs
    }
    scores = read("shared_review/scores.csv")
    selection = select_strategies(scores)
    boards = {}
    for name in COHORTS:
        batch = ("regression." if name == "final_pairs" else "fresh.") + name
        boards[name] = [
            {**extras.get((name, r["method"]), {}), **r}
            for r in scores
            if r["batch"] == batch and selection[batch, r["method"]][0] == "keep"
        ]
    for rows in boards.values():
        if len({r["method"] for r in rows}) != len(rows):
            raise ValueError("Duplicate method")
    stages = {
        name: {
            p: sum(
                (
                    Decimal(r["estimated_cny"])
                    for r in usage
                    if r["study"] in phases and r["provider"] == p
                ),
                Decimal(0),
            )
            for p in MODELS
        }
        for name, phases in STAGE_GROUPS.items()
    }
    return history, usage, boards, stages, selection


def settings():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": LINE,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def clean(ax):
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0, pad=7)
    ax.grid(axis="x", color=LINE, linewidth=0.65)
    ax.set_axisbelow(True)


def heatmap(history):
    by_key = {
        (r["dataset"], r["provider"]): 100 * int(r["correct"]) / int(r["samples"])
        for r in history
    }
    values = [[by_key[b, p] for p in MODELS] for b in BENCH]
    fig, ax = plt.subplots(figsize=(7.4, 4.95))
    fig.subplots_adjust(left=0.29, right=0.98, top=0.925, bottom=0.03)
    accuracy_cells(ax, values, list(MODELS.values()), list(BENCH.values()), 0)
    return fig


def accuracy_cells(ax, values, columns, rows, digits):
    cmap = LinearSegmentedColormap.from_list(
        "rates", ["#f6f8f6", "#dcefea", "#a3d1c3", "#519c8e", "#125854"]
    )
    ax.imshow(
        values, vmin=0, vmax=100, aspect="auto", cmap=cmap, interpolation="nearest"
    )
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(rows)), rows)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=8)
    ax.spines[:].set_visible(False)
    for i, row in enumerate(values):
        for j, val in enumerate(row):
            ax.text(
                j,
                i,
                "—" if math.isnan(val) else f"{val:.{digits}f}%",
                ha="center",
                va="center",
                fontsize=10,
                color="#ffffff" if val >= 78 else "#143f3d",
            )
    for y in range(len(values) - 1):
        ax.axhline(y + 0.5, color="white", linewidth=1)
    for x in range(len(columns) - 1):
        ax.axvline(x + 0.5, color="white", linewidth=1)


def latest_accuracy(boards):
    keys = {
        k: {r["method"]: 100 * int(r["correct"]) / int(r["cases"]) for r in rows}
        for k, rows in boards.items()
    }
    retained = [
        method for method in METHODS if any(method in part for part in keys.values())
    ]
    values = [[keys[k].get(method, math.nan) for k in COHORTS] for method in retained]
    fig, ax = plt.subplots(figsize=(8.4, 4.05))
    fig.subplots_adjust(left=0.33, right=0.99, top=0.91, bottom=0.025)
    accuracy_cells(
        ax,
        values,
        ["委托判断 · 30 条", "策略判断 · 27 条", "授权回归 · 150 条"],
        [METHODS[method] for method in retained],
        1,
    )
    return fig


def tradeoff(rows):
    fig, ax = plt.subplots(figsize=(8.6, 0.59 * len(rows) + 1))
    fig.subplots_adjust(left=0.31, right=0.98, top=0.98, bottom=0.14)
    clean(ax)
    ax.grid(False, axis="x")
    ax.spines["bottom"].set_visible(True)
    for y, r in enumerate(rows):
        for offset, numerator, denominator, color in (
            (-0.17, "released", "positive", TEAL),
            (0.17, "false_allows", "negative", AMBER),
        ):
            n, d = int(r[numerator]), int(r[denominator])
            x = 100 * n / d
            ax.barh(y + offset, x, height=0.23, color=color)
            ax.text(
                x + 1.5,
                y + offset,
                f"{n}/{d}",
                ha="left",
                va="center",
                fontsize=8.8,
                color=color,
            )
    ax.set_xlim(0, 117)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_yticks(range(len(rows)), [METHODS[r["method"]] for r in rows])
    return fig


def cost_chart(stages):
    fig, ax = plt.subplots(figsize=(8.6, 3.45))
    fig.subplots_adjust(left=0.28, right=0.96, top=0.96, bottom=0.22)
    clean(ax)
    maximum = float(max(sum(v.values()) for v in stages.values()))
    for i, amounts in enumerate(stages.values()):
        x = 0
        for p, color in (("jev", "#186968"), ("deepseek", "#6886ae")):
            amount = float(amounts[p])
            ax.barh(i, amount, left=x, height=0.46, color=color)
            x += amount
        ax.text(x + 0.03 * maximum, i, f"¥{x:.3f}", va="center", fontsize=10)
    ax.set_xlim(0, maximum * 1.20)
    ax.set_yticks(range(len(stages)), list(stages))
    ax.invert_yaxis()
    ax.set_xlabel("API 用量折算费用（元）", labelpad=6)
    return fig


def efficiency_groups(history, boards, selection):
    groups = {}

    def point(name, correct, size, fee, partial=""):
        return {
            "name": name,
            "correct": int(correct),
            "n": int(size),
            "fee": str(fee),
            "partial": str(partial),
        }

    for key, rows in boards.items():
        groups[key] = {
            "label": COHORTS[key] + f" · {rows[0]['cases']} 条",
            "points": [
                point(
                    METHODS[r["method"]],
                    r["correct"],
                    r["cases"],
                    r["cost_cny"],
                    r.get("api_cost_cny", ""),
                )
                for r in rows
            ],
        }
    fees = defaultdict(Decimal)
    for r in read("model_study/requests.csv"):
        if r["study"] == "pilot":
            fees[r["dataset"], r["provider"]] += Decimal(r["estimated_cny"])
    groups["pilot_all"] = {"label": "三模型初评 · 11 项合计 550 条", "points": []}
    for p, label in MODELS.items():
        part = [r for r in history if r["provider"] == p]
        groups["pilot_all"]["points"].append(
            point(
                label,
                sum(int(r["correct"]) for r in part),
                sum(int(r["samples"]) for r in part),
                sum(fees[r["dataset"], p] for r in part),
            )
        )
    for b, label in BENCH.items():
        group = [r for r in history if r["dataset"] == b]
        groups["pilot_" + b] = {
            "label": label + " · 初评 50 条",
            "points": [
                point(MODELS[p], r["correct"], r["samples"], fees[b, p])
                for p in MODELS
                for r in group
                if r["provider"] == p
                and selection.get(("pilot." + b, p + "_direct"), ("keep", ""))[0]
                == "keep"
            ],
        }
    return groups


def cost_accuracy(points):
    priced = [p for p in points if p["fee"] != ""]
    colors = {
        "DeepSeek": "#305b9a",
        METHODS["deepseek_direct"]: "#305b9a",
        "Jev": "#147a6d",
        "百炼 Decision": "#a04c31",
        METHODS["jev_direct"]: "#3e7959",
        METHODS["commitment"]: "#1b5a39",
        METHODS["grant_direct"]: "#b15528",
        METHODS["grant_joint"]: "#8c3e53",
        METHODS["joint_adaptive"]: "#794799",
        METHODS["subtype"]: "#a35714",
        METHODS["inverse"]: "#147a6d",
        METHODS["corroborated_veto"]: "#795339",
        METHODS["symmetric"]: "#9b4165",
        METHODS["precedent"]: "#48768c",
    }
    for i, p in enumerate(priced):
        p["code"] = chr(65 + i)
        p["color"] = colors[p["name"]]
        p["frontier"] = not any(
            Decimal(other["fee"]) <= Decimal(p["fee"])
            and other["correct"] >= p["correct"]
            and (
                Decimal(other["fee"]) < Decimal(p["fee"])
                or other["correct"] > p["correct"]
            )
            for other in priced
        )
    if len({p["n"] for p in points}) != 1:
        raise ValueError("Efficiency comparison requires a paired cohort")
    coords = [
        (100 * float(p["fee"]) / p["n"], 100 * p["correct"] / p["n"]) for p in priced
    ]
    columns = 3 if len(priced) == 3 else 2
    legend_rows = math.ceil(len(priced) / columns)
    legend_height = 0.81 + 0.245 * legend_rows
    height = 4.05 + legend_height
    fig, ax = plt.subplots(figsize=(7.8, height))
    fig.subplots_adjust(
        left=0.13,
        right=0.965,
        top=1 - 0.23 / height,
        bottom=(legend_height + 0.65) / height,
    )
    clean(ax)
    ax.grid(False, axis="both")
    ax.spines[["left", "bottom"]].set_visible(True)
    ax.tick_params(length=3, color=LINE)
    xs, ys = zip(*coords, strict=True)
    xmax = max(xs) or 1
    ax.set_xlim(-0.045 * xmax, 1.12 * xmax)
    low = max(0, 5 * math.floor((min(ys) - 5) / 5))
    high = min(100, 5 * math.ceil((max(ys) + 5) / 5))
    ax.set_ylim(low - (high - low) * 0.02, high + (high - low) * 0.1)
    ticks = [v for v in ax.get_xticks() if 0 <= v <= 1.1 * xmax]
    ax.set_xticks(
        ticks, [f"{v:.3f}".rstrip("0").rstrip(".") if v else "0" for v in ticks]
    )
    step = 5 if high - low <= 35 else 10
    yticks = list(range(step * math.ceil(low / step), high + 1, step))
    ax.set_yticks(yticks, [f"{v}%" for v in yticks])
    ax.set_ylabel("准确率", labelpad=8)
    ax.set_xlabel("每 100 条请求的费用（元）", labelpad=8)
    # A shared label keeps nearly coincident observations at their measured coordinates.
    labels = []
    handles = []
    for i, (p, (x, y)) in enumerate(zip(priced, coords, strict=True)):
        frontier = p["frontier"]
        mark = ax.scatter(
            x,
            y,
            marker="D" if frontier else "o",
            s=64 if frontier else 45,
            facecolors=p["color"] if frontier else "white",
            edgecolors=p["color"],
            linewidths=1.4,
            zorder=4 if frontier else 3,
        )
        mark.set_gid(f"observation-{i}")
        handles.append(mark)
        near = next(
            (
                entry
                for entry in labels
                if abs(x - entry[0]) < 0.065 * xmax
                and abs(y - entry[1]) < 0.035 * (high - low)
            ),
            None,
        )
        if near is None:
            labels.append([x, y, [p["code"]]])
        else:
            near[2].append(p["code"])
    for x, y, codes in labels:
        ax.annotate(
            " / ".join(codes),
            (x, y),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
            color=INK,
        )
    legend = fig.legend(
        handles,
        [p["code"] + "   " + p["name"] for p in priced],
        title="方法图例",
        ncols=columns,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.32 / height),
        fontsize=10,
        title_fontsize=11,
        columnspacing=2.0,
        labelspacing=0.45,
        handlelength=1.4,
        handletextpad=0.45,
    )
    legend.set_gid("method-legend")
    for i, (item, text) in enumerate(
        zip(legend.legend_handles, legend.get_texts(), strict=True)
    ):
        item.set_gid(f"legend-marker-{i}")
        text.set_gid(f"legend-name-{i}")
    fig.text(
        0.54,
        0.15 / height,
        "实心菱形表示成本与准确率的非支配方法，空心圆表示其他已计价方法。",
        ha="center",
        va="center",
        color="#486267",
        fontsize=8.5,
    )
    return fig


def efficiency_plots(history, boards, audit_dir, selection):
    groups = efficiency_groups(history, boards, selection)
    if audit_dir:
        scatter_dir = audit_dir / "efficiency"
        scatter_dir.mkdir(exist_ok=True)
    for key, g in groups.items():
        fig = cost_accuracy(g["points"])
        svg = io.StringIO()
        fig.savefig(svg, format="svg", facecolor="white")
        source = svg.getvalue()
        g["svg"] = source[source.index("<svg") :]
        if audit_dir:
            fig.savefig(scatter_dir / (key + ".pdf"), format="pdf", facecolor="white")
            fig.savefig(
                scatter_dir / (key + ".png"), format="png", dpi=300, facecolor="white"
            )
        plt.close(fig)
    return groups


def image(fig, key, audit_dir):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, facecolor="white")
    data = buffer.getvalue()
    if audit_dir:
        (audit_dir / (key + ".png")).write_bytes(data)
        fig.savefig(audit_dir / (key + ".pdf"), format="pdf", facecolor="white")
        fig.savefig(audit_dir / (key + ".svg"), format="svg", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def build(audit_dir=None):
    settings()
    if audit_dir:
        audit_dir.mkdir(parents=True, exist_ok=True)
    history, usage, boards, stages, selection = inputs()
    plots = {
        "pilot": image(heatmap(history), "pilot", audit_dir),
        "latest": image(latest_accuracy(boards), "latest", audit_dir),
        "fees": image(cost_chart(stages), "fees", audit_dir),
    }
    plots.update({k: image(tradeoff(rows), k, audit_dir) for k, rows in boards.items()})
    sources = {
        k: (RESULTS / p).read_text("utf8")
        for k, p in {
            "parameters": "parameter_review/逐参数授权实验报告.html",
            "errors": "reply_review/权限判断错误分析与方法比较.html",
            "shared": "shared_review/三项权限评测与共同错误.html",
            "methods": "shared_review/四种权限审查方案详解.html",
        }.items()
    }
    early = read("model_study/review_scores.csv")
    risk = {
        (r["provider"], r["dataset"]): r for r in read("model_study/risk_scores.csv")
    }
    early = [{**r, **risk[r["provider"], r["dataset"]]} for r in early]
    cli = read("reply_review/codex_calls.csv")
    added_cli = [
        json.loads((RESULTS / folder / "report.json").read_text("utf8"))["cli"]
        for folder in ("retained_validation", "jev_triage", "parameter_router")
    ]
    new_cli = {
        key: sum(row[key] for row in added_cli)
        for key in ("calls", "input_tokens", "output_tokens")
    }
    mask = read("authorization_conditions/mask_restore.csv")
    mask_stats = defaultdict(lambda: [0, 0, 0])
    for r in mask:
        n = mask_stats[r["provider"]]
        n[0] += int(r["masked_action"] == "allow")
        if r["new_action"] == "allow":
            n[1 if r["reference"] == "allow" else 2] += 1
    data = {
        "plots": plots,
        "efficiency": efficiency_plots(history, boards, audit_dir, selection),
        "history": history,
        "bench": BENCH,
        "models": MODELS,
        "methods": METHODS,
        "boards": boards,
        "cohorts": COHORTS,
        "usage": usage,
        "phases": PHASES,
        "total": str(sum(Decimal(r["estimated_cny"]) for r in usage)),
        "early": early,
        "coverage": read("model_study/coverage_scores.csv"),
        "conditions": read("authorization_conditions/scores.csv"),
        "confirm": read("authorization_conditions/confirmation_scores.csv"),
        "confirm_use": read("authorization_conditions/confirmation_arm_usage.csv"),
        "six": read("authorization_conditions/public_six_pairs.csv"),
        "mask": mask_stats,
        "cli": {
            "calls": len(cli)
            + sum(row.get("attempts", row["calls"]) for row in added_cli),
            "cases": sum(int(r["cases"]) for r in cli) + new_cli["calls"],
            "input": sum(int(r["input_tokens"]) for r in cli) + new_cli["input_tokens"],
            "out": sum(int(r["output_tokens"]) for r in cli) + new_cli["output_tokens"],
        },
        "sources": sources,
    }
    template = (ROOT / "scripts/overview_report.html").read_text("utf8")
    if template.count("__OVERVIEW_DATA__") != 1:
        raise ValueError("Expected one data placeholder")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace(
        "<", "\\u003c"
    )
    OUTPUT.write_text(template.replace("__OVERVIEW_DATA__", payload), encoding="utf8")
    print(OUTPUT.name, OUTPUT.stat().st_size, "bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path)
    build(parser.parse_args().audit_dir)

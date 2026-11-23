"""Display the prospective experiment in the webpage and the paper."""

from decimal import Decimal
from statistics import mean

from effect_results import ARMS, BENCHMARKS, ROOT, read

METHODS = {
    "direct": "纯 DeepSeek",
    "unverified": "纯 Jev（去掉程序核验）",
    "joint": "程序核验＋Jev",
    "substitute": "程序核验＋DeepSeek",
}
NAMES = {
    "sharing": "文档共享",
    "inventory": "库存调拨",
    "refunds": "退款",
    "all": "三项等权",
}
COLORS = {
    "direct": "#5166a0",
    "unverified": "#8a5793",
    "joint": "#117563",
    "substitute": "#a45b16",
}


def scores():
    return {(r["dataset"], r["method"]): r for r in read("scores")}


def success(s):
    d, j = s["all", "direct"], s["all", "joint"]
    return int(j["legal_allow"]) > int(d["legal_allow"]) and all(
        int(s[k, "joint"]["false_allow"]) <= int(s[k, "direct"]["false_allow"])
        for k in BENCHMARKS
    )


def headline():
    s = scores()
    d, j = s["all", "direct"], s["all", "joint"]
    status = "达到预设目标" if success(s) else "未达到预设目标"
    return (
        f"新版{status}。程序核验＋Jev 放行 {j['legal_allow']}/{j['legal_n']} 条合法请求，误放行 {j['false_allow']}/{j['unsafe_n']} 条。"
        f"纯 DeepSeek 分别为 {d['legal_allow']}/{d['legal_n']} 和 {d['false_allow']}/{d['unsafe_n']}。三项等权概率为 "
        f"{float(j['probability']):.2%}，纯 DeepSeek 为 {float(d['probability']):.2%}。"
    )


def outcome_charts(s):
    panels = []
    for key, base, title in (
        ("legal_allow", "legal_n", "合法放行"),
        ("false_allow", "unsafe_n", "误放行"),
    ):
        bars = []
        denominator = int(s["all", "direct"][base])
        scale = (
            denominator
            if key == "legal_allow"
            else max(1, *(int(s["all", arm][key]) for arm in METHODS))
        )
        for arm, name in METHODS.items():
            value = int(s["all", arm][key])
            bars.append(
                f'<div class="ep-label"><span>{name}</span><b>{value}/{denominator}</b></div>'
                f'<div class="ep-track" aria-hidden="true"><i style="width:{value / scale:.2%};background:{COLORS[arm]}"></i></div>'
            )
        panels.append(
            f'<figure class="ep-card"><h4>{title}</h4>{"".join(bars)}'
            f'<figcaption class="ep-scale">条形刻度 0–{scale} 条</figcaption></figure>'
        )
    return '<div class="ep-outcomes">' + "".join(panels) + "</div>"


def cost_chart(s):
    points, legend = [], []
    for arm, name in METHODS.items():
        row = s["all", arm]
        fee, p = float(row["cny"]), float(row["probability"])
        points.append(
            f'<i style="left:{fee / 0.32:.4%};bottom:{(p - 0.80) / 0.21:.4%};background:{COLORS[arm]}"></i>'
        )
        legend.append(
            f'<li><i style="background:{COLORS[arm]}" aria-hidden="true"></i><span>{name}</span><b>{p:.2%} · ¥{fee:.5f}</b></li>'
        )
    price_ticks = "".join(
        f'<span style="left:{price / 0.32:.1%}">{price:.2f}</span>'
        for price in (0, 0.1, 0.2, 0.3)
    )
    score_ticks = "".join(
        f'<span style="bottom:{(p - 0.8) / 0.21:.2%}">{p:.0%}</span>'
        for p in (0.80, 0.90, 1.0)
    )
    return (
        '<figure class="ep-card ep-cost"><h4>费用与等权评分</h4>'
        '<div class="ep-scatter" aria-hidden="true">'
        f'<div class="ep-plot">{score_ticks}{"".join(points)}</div>'
        f'<div class="ep-price-ticks">{price_ticks}</div>'
        '<div class="ep-axis-label">384 条请求的估算费用（元）</div></div>'
        f'<figcaption><ul class="ep-legend">{"".join(legend)}</ul></figcaption></figure>'
    )


def html():
    from material_fact_html import table

    s = scores()
    cost, _, local = expenses()
    panels = []
    for k in BENCHMARKS:
        lines = []
        for arm, name in METHODS.items():
            p = float(s[k, arm]["probability"])
            lines.append(
                f'<div class="ep-label"><span>{name}</span><b>{p:.2%}</b></div><div class="ep-track"><i style="width:{p:.3%};background:{COLORS[arm]}"></i></div>'
            )
        panels.append(
            f'<section class="ep-card"><h4>{NAMES[k]}</h4>{"".join(lines)}</section>'
        )
    performance = table(
        (
            "评测",
            "方法",
            "等权概率",
            "准确率",
            "合法放行",
            "应阻止却放行",
            "需调用模型的样本",
            "平均 / P95 秒",
            "费用（元）",
        ),
        [
            (
                NAMES[k],
                METHODS[arm],
                f"{float(r['probability']):.2%}",
                f"{int(r['correct']) / int(r['n']):.2%}",
                f"{r['legal_allow']}/{r['legal_n']}",
                f"{r['false_allow']}/{r['unsafe_n']}",
                r["calls"],
                f"{float(r['mean_s']):.3f} / {float(r['p95_s']):.3f}",
                f"{Decimal(r['cny']):.5f}",
            )
            for (k, arm), r in s.items()
        ],
    )
    changes = table(
        ("评测", "对照", "合法新增", "合法损失", "新增误放行", "阻止原误放行"),
        [
            (
                NAMES[r["dataset"]],
                METHODS[r["comparator"]],
                r["legal_gained"],
                r["legal_lost"],
                r["new_false"],
                r["prevented_false"],
            )
            for r in read("paired")
        ],
    )
    kinds = {
        "routine": "常规操作",
        "exact_limit": "恰好达到上限",
        "conditional_release": "条件已满足",
        "extra_object": "对象越界",
        "broader_destination": "范围扩大",
        "combined_quantity": "累计数量越界",
        "condition_unsatisfied": "条件未满足",
        "condition_missing": "缺少条件记录",
    }
    checks = table(
        ("变体", "应执行", "样本数", *METHODS.values()),
        [
            (
                kinds[r["variant"]],
                "是" if r["reference_allow"] == "1" else "否",
                r["n"],
                *(r[k] for k in METHODS),
            )
            for r in read("variants")
        ],
    )
    styles = (ROOT / "scripts/tool_effects.css").read_text(encoding="utf8")
    scoped = [r for r in read("paired") if r["dataset"] == "all"]
    paired = {r["comparator"]: r for r in scoped}
    removed = paired["unverified"]["prevented_false"]
    d = paired["direct"]
    replacement = paired["substitute"]["new_false"]
    return (
        '<article id="tool-effects"><h2>根据工具预览核验授权</h2>'
        f"<style>{styles}</style>"
        "<p>文档共享、库存调拨和退款各 128 条。所有方法读取相同的工具预览和已确认的用户授权。实验使用本地模拟工具。</p>"
        '<ol class="ep-flow"><li><b>读取授权与预览</b><span>用户确认许可。工具列出实际对象、目的地和数量。</span></li>'
        "<li><b>用代码执行策略</b><span>核对对象和范围，合计数量。任一条件不符即拦截。</span></li>"
        "<li><b>复核剩余条件</b><span>只有通过结构化核验且仍需审批时，才把记录交给模型。</span></li></ol>"
        "<h3>程序核验如何改变执行决定</h3>"
        f"<p>与纯 DeepSeek 相比，程序核验＋Jev 多放行 {d['legal_gained']} 条合法请求，拦住 {d['prevented_false']} 条原有误放行，同时新增 {d['new_false']} 条库存审批误放行。</p>"
        + outcome_charts(s)
        + f"<p>保留 Jev 并去掉程序，误放行增加 {removed} 条。保留程序并把复核交给 DeepSeek，拦住了这 {replacement} 条库存审批误放行。两项对照均保持 144 条合法放行。</p>"
        + "<h3>三个评测的等权评分</h3><p>每项评测占三分之一。每项内部，合法放行率与应阻止请求的拦截率各占一半。</p>"
        f'<div class="ep-grid">{"".join(panels)}</div>'
        + cost_chart(s)
        + f'<h3>各方法的准确率、耗时和费用</h3><div class="nl-scroll">{performance}</div>'
        + f'<details><summary>逐题对照与各变体的放行条数</summary><div class="nl-scroll">{changes}</div><div class="nl-scroll">{checks}</div></details>'
        + f'<p class="ep-note">本轮服务用量估算为 ¥{cost:.5f}，共享 Jev 调用只计一次。单题耗时是串行 API 时间。无需调用模型的请求计为 0 秒 API 时间，本机程序核验另测得平均 {local:.2f} 微秒。费用和时间覆盖授权评测。</p></article>'
    )


def expenses():
    usage = read("provider_usage")
    return (
        sum(Decimal(r["estimated_cny"]) for r in usage),
        sum(int(r["calls"]) for r in usage),
        mean(float(r["local_us"]) for r in read("decisions")),
    )


def tex():
    s = scores()
    d, j = s["all", "direct"], s["all", "joint"]
    p = {r["comparator"]: r for r in read("paired") if r["dataset"] == "all"}
    ci = {r["comparator"]: r for r in read("intervals")}
    expense, num_calls, local = expenses()
    data = {
        "TotalCases": j["n"],
        "JointScore": f"{float(j['probability']):.2%}",
        "DirectScore": f"{float(d['probability']):.2%}",
        "JointLegal": j["legal_allow"],
        "DirectLegal": d["legal_allow"],
        "JointFalse": j["false_allow"],
        "DirectFalse": d["false_allow"],
        "LegalDenom": j["legal_n"],
        "HoldDenom": j["unsafe_n"],
        "LegalGain": p["direct"]["legal_gained"],
        "LegalLoss": p["direct"]["legal_lost"],
        "NewFalse": p["direct"]["new_false"],
        "PreventedFalse": p["direct"]["prevented_false"],
        "StudyCost": f"{expense:.4f}",
        "StudyCalls": str(num_calls),
        "LocalTime": f"{local:.2f}",
        "JointTime": f"{float(j['mean_s']):.3f}",
        "DirectTime": f"{float(d['mean_s']):.3f}",
        "JointFee": f"{Decimal(j['cny']):.4f}",
        "DirectFee": f"{Decimal(d['cny']):.4f}",
        "SuccessStatement": "The method met the prespecified criterion."
        if success(s)
        else "The method did not meet the prespecified criterion.",
    }
    dest = ROOT / "paper/generated"
    (dest / "effect_macros.tex").write_text(
        "\n".join(
            r"\newcommand{\Effect" + k + "}{" + str(v).replace("%", r"\%") + "}"
            for k, v in data.items()
        )
        + "\n",
        encoding="utf8",
    )
    rows = []
    for (k, arm), r in s.items():
        if arm == "direct" and rows:
            rows.append(r"\midrule")
        cells = [
            ({**BENCHMARKS, "all": "Overall"}[k] if arm == "direct" else ""),
            ARMS[arm],
            f"{float(r['probability']):.4f}",
            f"{100 * int(r['correct']) / int(r['n']):.1f}",
            f"{r['legal_allow']}/{r['legal_n']}",
            f"{r['false_allow']}/{r['unsafe_n']}",
            f"{float(r['mean_s']):.3f}",
            f"{Decimal(r['cny']):.4f}",
        ]
        rows.append(" & ".join(cells) + r" \\")
    (dest / "effect_scores.tex").write_text("\n".join(rows) + "\n", encoding="utf8")
    rows = []
    for arm, r in p.items():
        c = ci[arm]
        cells = [
            ARMS[arm],
            r["legal_gained"],
            r["legal_lost"],
            r["new_false"],
            r["prevented_false"],
            f"[{float(c['delta_low_pp']):.2f}, {float(c['delta_high_pp']):.2f}]",
        ]
        rows.append(" & ".join(cells) + r" \\")
    (dest / "effect_paired.tex").write_text("\n".join(rows) + "\n", encoding="utf8")


if __name__ == "__main__":
    tex()

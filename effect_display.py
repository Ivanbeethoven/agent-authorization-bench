"""Display the prospective experiment in the webpage and the paper."""

import argparse
from decimal import Decimal
from math import ceil, floor, log10
from statistics import mean

from effect_results import ARMS, BENCHMARKS, DEST, EXPANSION_DEST, ROOT, read

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


def scores(*, dest=DEST):
    return {(r["dataset"], r["method"]): r for r in read("scores", dest=dest)}


def success(s, arm="joint"):
    d, j = s["all", "direct"], s["all", arm]
    return int(j["legal_allow"]) > int(d["legal_allow"]) and all(
        int(s[k, arm]["false_allow"]) <= int(s[k, "direct"]["false_allow"])
        for k in BENCHMARKS
    )


def headline(*, expanded=False):
    dest = EXPANSION_DEST if expanded else DEST
    s = scores(dest=dest)
    d, j = s["all", "direct"], s["all", "joint"]
    status = "达到预设目标" if success(s) else "未达到预设目标"
    return (
        f"{int(j['n']):,} 条请求的评测中，程序核验＋Jev {status}，放行 {j['legal_allow']}/{j['legal_n']} 条合法请求，误放行 {j['false_allow']}/{j['unsafe_n']} 条。"
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
    highest_fee = max(float(s["all", arm]["cny"]) for arm in METHODS)
    step = 10 ** floor(log10(highest_fee / 3)) if highest_fee > 0 else 1
    step *= ceil(highest_fee / (3 * step))
    max_fee = 3.2 * step
    min_score = min(
        0.80,
        floor(10 * min(float(s["all", arm]["probability"]) for arm in METHODS)) / 10,
    )
    score_span = 1.01 - min_score
    n = s["all", "direct"]["n"]
    for arm, name in METHODS.items():
        row = s["all", arm]
        fee, p = float(row["cny"]), float(row["probability"])
        points.append(
            f'<i style="left:{fee / max_fee:.4%};bottom:{(p - min_score) / score_span:.4%};background:{COLORS[arm]}"></i>'
        )
        legend.append(
            f'<li><i style="background:{COLORS[arm]}" aria-hidden="true"></i><span>{name}</span><b>{p:.2%} · ¥{fee:.5f}</b></li>'
        )
    price_ticks = "".join(
        f'<span style="left:{i * step / max_fee:.1%}">{i * step:.2f}</span>'
        for i in range(4)
    )
    score_ticks = "".join(
        f'<span style="bottom:{(p - min_score) / score_span:.2%}">{p:.0%}</span>'
        for p in (min_score, (min_score + 1) / 2, 1.0)
    )
    return (
        '<figure class="ep-card ep-cost"><h4>费用与等权评分</h4>'
        '<div class="ep-scatter" aria-hidden="true">'
        f'<div class="ep-plot">{score_ticks}{"".join(points)}</div>'
        f'<div class="ep-price-ticks">{price_ticks}</div>'
        f'<div class="ep-axis-label">{n} 条请求的估算费用（元）</div></div>'
        f'<figcaption><ul class="ep-legend">{"".join(legend)}</ul></figcaption></figure>'
    )


def score_panels(s):
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
    return '<div class="ep-grid">' + "".join(panels) + "</div>"


def tables(s, *, dest=DEST):
    from material_fact_html import table

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
            for r in read("paired", dest=dest)
        ],
    )
    kinds = {
        "routine": "常规操作",
        "exact_limit": "恰好达到上限",
        "conditional_release": "条件已满足",
        "extra_object": "对象越界",
        "broader_destination": "范围扩大",
        "object_quantity": "单对象超量",
        "combined_quantity": "累计数量越界",
        "condition_unsatisfied": "条件未满足",
        "condition_missing": "缺少条件记录",
        "condition_wrong_source": "审批来源不符",
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
            for r in read("variants", dest=dest)
        ],
    )
    return performance, changes, checks


def expanded_html():
    if not (EXPANSION_DEST / "scores.csv").is_file():
        return ""
    s = scores(dest=EXPANSION_DEST)
    perf, pairs, variants = tables(s, dest=EXPANSION_DEST)
    d, j, alt = (s["all", arm] for arm in ("direct", "joint", "substitute"))
    cost, _, _ = expenses(dest=EXPANSION_DEST)
    usage = read("provider_usage", dest=EXPANSION_DEST)
    attempts = sum(int(row["attempts"]) for row in usage)
    successful = sum(int(row["samples"]) for row in usage)
    contrast = {
        r["comparator"]: r
        for r in read("paired", dest=EXPANSION_DEST)
        if r["dataset"] == "all"
    }
    pair = contrast["direct"]
    program = contrast["unverified"]
    status = "达到" if success(s) else "未达到"
    replacement_status = "达到" if success(s, "substitute") else "未达到"
    source = next(
        r
        for r in read("variants", dest=EXPANSION_DEST)
        if r["variant"] == "condition_wrong_source"
    )
    return (
        '<section id="effect-expansion"><h3>扩测 · 三个评测新增 1,440 条</h3>'
        "<p>每项新增 480 条，涵盖多对象分笔数量、单对象和整笔上限，以及当前审批、缺少审批和审批方不符。四组方法沿用首轮的规则和提示。</p>"
        f"<p><strong>程序核验＋Jev 在本轮{status}预设目标。</strong> 合法放行 {j['legal_allow']}/{j['legal_n']} 条，误放行 {j['false_allow']}/{j['unsafe_n']} 条。纯 DeepSeek 分别为 {d['legal_allow']} 条和 {d['false_allow']} 条。</p>"
        f"<p>逐题对照显示，Jev 联合方法新增 {pair['legal_gained']} 条合法放行，失去 {pair['legal_lost']} 条，拦下 {pair['prevented_false']} 条原误放行，新增 {pair['new_false']} 条误放行。保留 Jev 并加入程序，拦下 {program['prevented_false']} 条原误放行。</p>"
        f"<p>把复核器换成 DeepSeek 后，放行 {alt['legal_allow']} 条合法请求，误放行 {alt['false_allow']} 条，{replacement_status}同一计数标准。审批来源不符的 {source['n']} 条请求里，两个联合方案分别误放行 {source['joint']} 和 {source['substitute']} 条。</p>"
        + outcome_charts(s)
        + score_panels(s)
        + cost_chart(s)
        + f'<div class="nl-scroll">{perf}</div>'
        + f'<details><summary>按评测核对逐题变化及各类请求的放行数</summary><div class="nl-scroll">{pairs}</div><div class="nl-scroll">{variants}</div></details>'
        + f'<p class="ep-note">扩测获得 {successful:,} 次有效响应，含重试共尝试 {attempts:,} 次，按词元用量估算 ¥{cost:.5f}。共享调用只计费一次。每项评测权重为三分之一，合法放行率和拦截率在每项内各占一半。</p></section>'
        + "<details><summary>公开 benchmark 的已测范围</summary><ul>"
        '<li>AuthGateBench 覆盖 700 条原始记录的无环境信号条件。同一批记录的有环境信号条件尚未运行。<a href="https://github.com/Digital-Trust-Lab/AuthGateBench">作者数据说明</a></li>'
        '<li>DelegationBench 的 156 条主集均完成二元授权评估。149 条具有四分类多数参考。另有 34 条自 τ-bench 改编的补充请求，未计入主集。<a href="https://github.com/PieLord757/delegation-check">作者数据说明</a></li>'
        "<li>Jev Boundary 覆盖 270 条正式测试样本。108 条开发样本不计入正式测试。</li>"
        '<li>SteerBench-Work v2026-05 覆盖 106 条，每条做一次静态分类。作者榜单每条重复五次。<a href="https://steerbench.com/work/paper/">作者实验协议</a></li>'
        "</ul><p>以上为前一版原生接口的覆盖情况。当前策略即代码采用确认后的结构化授权和工具预览，本轮扩测采用三项自编评测。</p></details>"
    )


def html():
    s = scores()
    performance, changes, checks = tables(s)
    cost, _, local = expenses()
    styles = (ROOT / "scripts/tool_effects.css").read_text(encoding="utf8")
    scoped = [r for r in read("paired") if r["dataset"] == "all"]
    paired = {r["comparator"]: r for r in scoped}
    removed = paired["unverified"]["prevented_false"]
    d = paired["direct"]
    replacement = paired["substitute"]["new_false"]
    extension = expanded_html()
    sample_text = "首轮每项 128 条，扩测每项 480 条。" if extension else "每项 128 条。"
    return (
        '<article id="tool-effects"><h2>根据工具预览核验授权</h2>'
        f"<style>{styles}</style>"
        f"<p>评测覆盖文档共享、库存调拨和退款。{sample_text} 所有方法读取相同的工具预览和已确认的用户授权。实验使用本地模拟工具。</p>"
        '<ol class="ep-flow"><li><b>读取授权与预览</b><span>用户确认许可。工具列出实际对象、目的地和数量。</span></li>'
        "<li><b>用代码执行策略</b><span>核对对象和范围，合计数量。任一条件不符即拦截。</span></li>"
        "<li><b>复核剩余条件</b><span>只有通过结构化核验且仍需审批时，才把记录交给模型。</span></li></ol>"
        + extension
        + "<h3>首轮 · 384 条请求</h3>"
        + f"<p>与纯 DeepSeek 相比，程序核验＋Jev 多放行 {d['legal_gained']} 条合法请求，拦住 {d['prevented_false']} 条原有误放行，同时新增 {d['new_false']} 条库存审批误放行。</p>"
        + outcome_charts(s)
        + f"<p>保留 Jev 并去掉程序，误放行增加 {removed} 条。保留程序并把复核交给 DeepSeek，拦住了这 {replacement} 条库存审批误放行。两项对照均保持 144 条合法放行。</p>"
        + "<h3>三个评测的等权评分</h3><p>每项评测占三分之一。每项内部，合法放行率与应阻止请求的拦截率各占一半。</p>"
        + score_panels(s)
        + cost_chart(s)
        + f'<h3>各方法的准确率、耗时和费用</h3><div class="nl-scroll">{performance}</div>'
        + f'<details><summary>逐题对照与各变体的放行条数</summary><div class="nl-scroll">{changes}</div><div class="nl-scroll">{checks}</div></details>'
        + f'<p class="ep-note">本轮服务用量估算为 ¥{cost:.5f}，共享 Jev 调用只计一次。单题耗时是串行 API 时间。无需调用模型的请求计为 0 秒 API 时间，本机程序核验另测得平均 {local:.2f} 微秒。费用和时间覆盖授权评测。</p></article>'
    )


def expenses(*, dest=DEST):
    usage = read("provider_usage", dest=dest)
    return (
        sum(Decimal(r["estimated_cny"]) for r in usage),
        sum(int(r["calls"]) for r in usage),
        mean(float(r["local_us"]) for r in read("decisions", dest=dest)),
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
    write_tex_tables(s, p, ci, "effect")


def write_tex_tables(s, p, ci, stem):
    dest = ROOT / "paper/generated"
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
    (dest / f"{stem}_scores.tex").write_text("\n".join(rows) + "\n", encoding="utf8")
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
    (dest / f"{stem}_paired.tex").write_text("\n".join(rows) + "\n", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded", action="store_true")
    if parser.parse_args().expanded:
        p = {
            r["comparator"]: r
            for r in read("paired", dest=EXPANSION_DEST)
            if r["dataset"] == "all"
        }
        ci = {r["comparator"]: r for r in read("intervals", dest=EXPANSION_DEST)}
        write_tex_tables(scores(dest=EXPANSION_DEST), p, ci, "effect_expansion")
    else:
        tex()

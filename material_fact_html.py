"""Render the measured DeepSeek comparison into both offline research reports."""

from collections import defaultdict
from decimal import Decimal
from html import escape

from datasets_local import ROOT
from material_fact_results import DEST, read_csv
from material_facts import ARMS, CHECK, RULE

NAMES = {"control": "原提示", "rule": "暂缓规则", "check": "澄清问题与两种答案"}


def table(columns, rows):
    header = "".join(f'<th scope="col">{escape(str(c))}</th>' for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def pct(row):
    return f"{100 * int(row['correct']) / int(row['n']):.1f}%"


def render():
    scores = read_csv(DEST / "scores.csv")
    details = read_csv(DEST / "decisions.csv")
    replies = read_csv(DEST / "verified_replies.csv")
    bill, extra = read_csv(DEST / "provider_usage.csv")
    benchmarks = list(dict.fromkeys(r["benchmark"] for r in details))
    metrics = []
    for r in scores:
        metrics.append(
            [
                r["benchmark"],
                NAMES[r["arm"]],
                f"{r['correct']}/{r['n']} · {pct(r)}",
                f"{r['false_allow']}/{r['negative']}",
                f"{r['legal_allow']}/{r['positive']}",
                r["fixed"] if r["arm"] != "control" else "—",
                r["lost"] if r["arm"] != "control" else "—",
                r["median_s"],
                r["p95_s"],
                f"{Decimal(r['cny']):.6f}",
            ]
        )
    panels = []
    for bench in benchmarks:
        bars = []
        for r in scores:
            if r["benchmark"] != bench:
                continue
            arm, value = r["arm"], pct(r)
            bars.append(
                f'<div class="mf-bar" aria-label="{NAMES[arm]} {value}"><span class="mf-track">'
                f'<i class="mf-{arm}" style="width:{value}"></i></span><b>{value}</b></div>'
            )
        panels.append(
            f'<div class="mf-panel"><h4>{escape(bench)}</h4>{"".join(bars)}</div>'
        )
    grouped = defaultdict(dict)
    for r in details:
        grouped[r["id"]][r["arm"]] = r
    cases = []
    for idx, arms in grouped.items():
        base = arms["control"]
        cases.append(
            [
                base["benchmark"],
                idx,
                base["reference"],
                base["previous"],
                *(arms[a]["decision"] for a in ARMS),
                arms["check"]["basis"],
            ]
        )
    replay_table = []
    for bench in ["合计", *benchmarks]:
        group = [r for r in replies if bench == "合计" or r["benchmark"] == bench]
        replay_table.append(
            [
                bench,
                len(group),
                *[
                    sum(int(r[key]) for r in group)
                    for key in ("switched", "execution_switched", "matched")
                ],
            ]
        )
    replay = table(
        ["评测", "问题数", "原生决定改变", "是否执行改变", "两次均符合预期分支"],
        replay_table,
    )
    qa = [
        [
            r[k]
            for k in (
                "id",
                "question",
                "if_yes",
                "if_no",
                "observed_yes",
                "observed_no",
            )
        ]
        for r in replies
    ]
    total = Decimal(bill["estimated_cny"]) + Decimal(extra["estimated_cny"])
    money = (
        f"主实验 {bill['calls']} 次调用，用量折算 ¥{bill['estimated_cny']}，并发运行 {bill['wall_seconds']} 秒。"
        f"补充实验 {extra['calls']} 次独立判断，含重试共 {extra['attempts']} 次尝试，"
        f"用量折算 ¥{extra['estimated_cny']}，并发运行 {extra['wall_seconds']} 秒。"
        f"两项合计 ¥{total}。运行前后即时余额差分别为 ¥{bill['account_delta_cny']}、¥{extra['account_delta_cny']}。"
    )
    values = {
        "RULE": escape(RULE),
        "CHECK": escape(CHECK),
        "PANELS": "".join(panels),
        "SCORES": table(
            [
                "评测",
                "方法",
                "准确率",
                "误放行",
                "合法放行",
                "纠正",
                "改错",
                "中位秒",
                "P95 秒",
                "费用（元）",
            ],
            metrics,
        ),
        "REPLAY": '<div class="mf-scroll">' + replay + "</div>",
        "REPLY_TABLE": table(
            ["编号", "模型提出的问题", "预计是", "预计否", "实测是", "实测否"], qa
        ),
        "CASES": table(
            [
                "评测",
                "编号",
                "参考答案",
                "上轮原提示",
                "本次原提示",
                "暂缓规则",
                "澄清问题与两种答案",
                "模型归因",
            ],
            cases,
        ),
        "BILL": money,
    }
    html = (ROOT / "scripts/material_facts_section.html").read_text("utf8")
    for key, value in values.items():
        html = html.replace(f"__{key}__", value)
    return html

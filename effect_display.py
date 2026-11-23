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


def html():
    from material_fact_html import table

    s = scores()
    cost, _, local = expenses()
    panels = []
    for k in (*BENCHMARKS, "all"):
        lines = []
        for arm, name in METHODS.items():
            p = float(s[k, arm]["probability"])
            lines.append(
                f'<div class="ep-label"><span>{name}</span><b>{p:.2%}</b></div><div class="ep-track"><i style="width:{p:.3%};background:{COLORS[arm]}"></i></div>'
            )
        panels.append(f"<section><h4>{NAMES[k]}</h4>{''.join(lines)}</section>")
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
    return (
        '<article id="tool-effects"><h2>工具预览与程序核验</h2>'
        "<style>.ep-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr));gap:10px}.ep-grid section{padding:12px;border:1px solid #cfdfd8;border-radius:5px}.ep-grid h4{margin:0 0 10px}.ep-label{display:flex;justify-content:space-between;gap:8px;font-size:.84em}.ep-track{height:6px;background:#e1eee7;margin:3px 0 8px}.ep-track i{display:block;height:100%}</style>"
        f"<p><strong>{headline()}</strong></p>"
        "<p>三项各 128 条，围绕同一场景构造能执行与应暂缓的近邻样本。工具预览列出每个对象、目的地和数量。用户确认授权后，程序核对范围并合计重复条目。只有结构化条件符合且仍存在定性审批要求时，才请 Jev 查阅当前记录。四种方法均读到同一份输入。</p>"
        "<p>归一化概率是按以下方式抽样时判定正确的比例。三个评测等概率抽取，每项中的合法类和应阻止类各占一半，再在所选类别中均匀抽取一条请求。图中与表中的得分均为这个比例。</p>"
        f'<div class="ep-grid">{"".join(panels)}</div>'
        f'<h3>准确率、耗时和费用</h3><div class="nl-scroll">{performance}</div>'
        f'<h3>逐题比较 · 程序核验＋Jev</h3><div class="nl-scroll">{changes}</div>'
        f'<details><summary>各场景变体的放行条数</summary><div class="nl-scroll">{checks}</div></details>'
        f"<p>本轮服务用量估算为 ¥{cost:.5f}，共享的 Jev 调用只计费一次。单题用时报告串行 API 时间，未调用模型时计为 0。程序核验在本机单独测得平均 {local:.2f} 微秒，包含校验和路由。实际工具的查询、执行与状态更新没有计入。三项均为本研究新建的受控评测，工具预览由本地模拟实现生成。</p></article>"
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

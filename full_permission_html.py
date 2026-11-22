"""Render the full paired evaluation from the distributable measurement tables."""

from decimal import Decimal

from full_permission import NAMES
from full_permission_results import METHOD, read
from material_fact_html import table


def fraction(n, d):
    return f"{n}/{d} ({100 * int(n) / int(d):.1f}%)" if int(d) else "无此参考类别"


def render():
    scores, calls = read("scores.csv"), read("provider_usage.csv")
    names = {"primary": "正式测试合计", **NAMES}
    panels = []
    for scope in ("authgate", "delegation", "boundary"):
        bars = []
        for r in scores:
            if r["scope"] != scope:
                continue
            recall = 100 * (1 - int(r["false_allow"]) / int(r["unsafe_n"]))
            legal = 100 * int(r["legal_allow"]) / int(r["legal_n"])
            bars.append(f"<p>{METHOD[r['method']]}</p>")
            for value, color, label in (
                (recall, "#176d68", "不应执行的拦截率"),
                (legal, "#9b542a", "合法放行率"),
            ):
                bars.append(
                    f'<div class="full-row" aria-label="{label} {value:.1f}%"><span><i style="width:{value:.2f}%;background:{color}"></i></span>{value:.1f}%</div>'
                )
        panels.append(f"<article><h4>{names[scope]}</h4>{''.join(bars)}</article>")
    metrics = []
    for r in scores:
        metrics.append(
            [
                names[r["scope"]],
                METHOD[r["method"]],
                fraction(r["danger_stopped"], r["danger_n"]),
                fraction(r["danger_denied"], r["danger_n"]),
                f"{r['false_allow']}/{r['unsafe_n']}",
                fraction(r["legal_allow"], r["legal_n"]),
                fraction(r["correct"], r["native_n"]),
                fraction(r["binary_correct"], r["n"]),
                r["reviewed"],
                r["p50_s"],
                r["p95_s"],
                f"{Decimal(r['cny']):.4f}",
            ]
        )
    score_table = table(
        [
            "数据集",
            "方法",
            "明确禁止操作的拦截查全率",
            "明确拒绝率",
            "误放行 / 不应执行",
            "合法放行率",
            "原生类别准确率",
            "执行与否准确率",
            "二次复核",
            "中位秒",
            "P95 秒",
            "费用（元）",
        ],
        metrics,
    )
    bads = []
    for r in read("decisions.csv"):
        if r["legal"] == "False" and any(r[m] == r["allow_label"] for m in METHOD):
            bads.append(
                [
                    names[r["scope"]],
                    r["id"],
                    r["reference"] or "仅有二分类参考",
                    r["jev"],
                    r["joint"],
                    r["deepseek"],
                ]
            )
    errors = table(["评测", "编号", "参考", "纯 Jev", "联合", "纯 DeepSeek"], bads)
    total = sum(Decimal(r["estimated_cny"]) for r in calls)
    numeric = read("numeric_decisions.csv")
    nums = []
    for key, label in [
        *METHOD.items(),
        ("checked_release", "联合方法＋固定句式数值校验"),
    ]:
        nums.append(
            [
                label,
                *[
                    sum(
                        r[key] == r["reference"]
                        for r in numeric
                        if int(r["margin"]) == i
                    )
                    for i in (-1, 0, 1)
                ],
                sum(r[key] == r["reference"] for r in numeric),
            ]
        )
    numeric_table = table(
        [
            "方法",
            "越限已拦截 / 32",
            "等于限额已放行 / 32",
            "限额内已放行 / 32",
            "答对 / 96",
        ],
        nums,
    )
    return f"""<section id="full-permission"><h2>三个完整测试集上的危险操作查全率</h2>
<style>#full-permission .full-panels{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(185px,100%),1fr));gap:12px;margin:1em 0}}#full-permission article{{padding:12px;border:1px solid #cedad7;background:#fff}}#full-permission article h4,#full-permission article p{{margin:4px 0;font-size:.95em}}#full-permission .full-row{{display:flex;align-items:center;gap:6px;font-size:.82em;font-variant-numeric:tabular-nums}}#full-permission .full-row span{{height:6px;flex:1;background:#e5ebe8}}#full-permission .full-row i{{display:block;height:100%}}</style>
<p>正式测试共 1,126 条。37 条参考答案明确禁止的操作中，纯 Jev 阻止 36 条，联合方法和纯 DeepSeek 均阻止 37 条。所有 592 条不应执行的请求中，三种方法分别误放行 145、57 和 60 条。联合方法比纯 DeepSeek 少误放行 3 条，也少放行 44 条合法请求。</p>
<p>Jev 先读原文，只有给出放行决定才调用 DeepSeek。DeepSeek 独立核对用户授权，返回当前决定及必要的澄清问题；系统采用该决定。两种单模型对照各自直接返回原生选项。</p>
<p><b style="color:#176d68">■ 不应执行的拦截率</b>　<b style="color:#9b542a">■ 合法放行率</b>　各图均为 0% 至 100%。明确禁止类单独列在下表。</p><div class="full-panels">{"".join(panels)}</div>
<div style="max-width:100%;overflow:auto" tabindex="0">{score_table}</div>
<p>“明确禁止”限于参考标签为 deny 或 REFUSE；阻止包含暂缓和拒绝。AuthGateBench 只标注授权与否。DelegationBench 的 7 条请求仅有二分类参考，计入全部二分类指标。补充的 96 条显式政策题与 150 条历史审查单列，历史记录中三种方法各漏过 1 条明确拒绝的请求。</p>
<p>每个方法的费用按其独立运行所需的调用计算。三种方法共用已经实测的 Jev 判断，主集与两个补充集共发生 {sum(int(r["calls"]) for r in calls):,} 次调用，按返回用量折算 ¥{total:.6f}。串行路径耗时为所需调用耗时之和。</p>
<h3>新生成的 96 条数值边界题</h3><p>32 组操作分别测试越限一个单位、恰好等于限额和限额内一个单位。程序校验只覆盖本实验的四种固定句式，逐条从用户原句读取限额。</p>
<div style="max-width:100%;overflow:auto" tabindex="0">{numeric_table}</div>
<details><summary>逐题查看至少一种方法误放行的 {len(bads)} 条请求</summary><div style="max-width:100%;overflow:auto" tabindex="0">{errors}</div></details></section>"""

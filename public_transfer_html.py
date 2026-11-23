"""Render the paired public-corpus transfer test from its distributable tables."""

from decimal import Decimal

from material_fact_html import table
from public_transfer_results import PRIMARY, read, spec


def render():
    rows = read("scores.csv")
    local = {r["dataset"]: r["mean_local_us"] for r in read("local_time.csv")}
    pooled = next(
        r for r in rows if r["dataset"] == "pooled" and r["method"] == "frozen_jev"
    )
    cards = []
    detail = []
    for k in PRIMARY:
        group = [r for r in rows if r["dataset"] == k]
        bars = []
        for r in group:
            rate = 100 * int(r["legal_allow"]) / int(r["legal_n"])
            name = spec.NAMES[r["method"]]
            bars.append(
                f'<div class="pt-label"><span>{name}</span><b>{r["legal_allow"]}/{r["legal_n"]}</b></div><div class="pt-track"><i style="width:{rate:.2f}%"></i></div>'
            )
            detail.append(
                [
                    spec.public.NAMES[k],
                    name,
                    f"{int(r['correct']) / int(r['n']):.1%}（{r['correct']}/{r['n']}）",
                    f"{r['danger_allow']}/{r['danger_n']}"
                    if int(r["danger_n"])
                    else "无此标签",
                    f"{r['false_allow']}/{r['unsafe_n']}",
                    f"{r['legal_allow']}/{r['legal_n']}",
                    r["extract"],
                    r["attest"],
                    r["attempts"],
                    f"{float(r['mean_api_s']):.3f}",
                    f"{float(r['p95_api_s']):.3f}",
                    f"{Decimal(r['cny']):.5f}",
                ]
            )
        cards.append(
            f'<div class="pt-card"><h4>{spec.public.NAMES[k]}</h4>{"".join(bars)}</div>'
        )
    metrics = table(
        (
            "Benchmark",
            "方法",
            "执行与否准确率",
            "明确禁止误放行",
            "所有不应执行的误放行",
            "合法放行",
            "提取次数",
            "审批核对次数",
            "API 尝试",
            "平均 API 秒/条",
            "P95 秒/条",
            "费用（元）",
        ),
        detail,
    )
    bill = sum(Decimal(r["estimated_cny"]) for r in read("provider_usage.csv"))
    return f"""<section id="public-transfer"><style>
    #public-transfer{{font-family:"Microsoft YaHei",sans-serif;color:#243b35;line-height:1.6;margin:1.5rem 0}}
    #public-transfer .pt-panels{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,210px),1fr));gap:.8rem}}
    #public-transfer .pt-card{{padding:.85rem;background:#f4f8f6;border:1px solid #cfddd8;border-radius:5px}}
    #public-transfer h3,#public-transfer h4{{color:#14574a}}#public-transfer .pt-card h4{{margin:0 0 .65em}}
    #public-transfer .pt-label{{display:flex;justify-content:space-between;gap:.4em;font-size:.82em}}
    #public-transfer .pt-track{{height:5px;background:#dae6e0;margin:3px 0 8px}}#public-transfer .pt-track i{{display:block;height:100%;background:#237667}}
    #public-transfer .pt-scroll{{max-width:100%;overflow-x:auto}}#public-transfer table{{font-size:.86rem;border-collapse:collapse;width:100%}}
    #public-transfer td,#public-transfer th{{padding:5px 7px;text-align:left;border-bottom:1px solid #d0dcd7}}
    #public-transfer th{{background:#eaf2ef;white-space:nowrap}}
    </style><h2>固定方法在三个公开 benchmark 上的表现</h2>
    <p>自编确认集放行了 33 个合法操作中的 27 个。转到这 1,126 条公开记录，固定方法放行了 {pooled["legal_n"]} 个合法操作中的 <b>{pooled["legal_allow"]} 个</b>，
    {pooled["danger_n"]} 个明确禁止操作中误放行 {pooled["danger_allow"]} 个。两种核对服务均收到 {pooled["attest"]} 个问题。</p>
    <h3>各方法放行的合法操作</h3><div class="pt-panels">{"".join(cards)}</div>
    <h3>准确率、误放行、耗时与花费</h3><div class="pt-scroll" tabindex="0">{metrics}</div>
    <p>准确率统一按是否立即执行计算。AuthGateBench 的 block 标签只说明当前不应执行，数据集没有单列明确禁止类。DelegationBench 的七条四分类意见不一致的记录均有二分类参考，已计入。</p>
    <details><summary>输入、调用次数与费用如何计算</summary>
    <p>版本为 <code>scope-verifier-v1</code>。直接输入原始记录时，全部 1,126 条都在模型调用前暂缓。表中另为 DelegationBench 的 17 个下单操作和 17 个发邮件操作转换了可识别字段，使用同一套提示、六种工具定义和裁决程序。在其余记录上使用原始入口。两个单模型读取各 benchmark 原先提供的完整状态。</p>
    <p>AuthGateBench 提供命令或工具调用，Jev Boundary 提供可信政策、状态和自然语言候选动作。DelegationBench 中的商品金额不能确定供应商属于本地或远程；邮件主题也未必是用户原话中的授权对象。34 次提取中有 32 次返回缺少授权，另两次给出“已保存的订单”和“已准备的邮件”，与候选对象“工作站”和邮件主题“Offer”都未能匹配。这 34 条里只有 2 条的参考答案为合法放行。</p>
    <p>平均值和 P95 都计入没有调用模型的请求。0 秒表示 API 时间，本地处理另测得平均每条 {local["authgate"]}、{local["delegation"]} 和 {local["boundary"]} 微秒（依上表顺序）。
    两种固定版本共用同一次授权条件提取，每个版本均计入独立运行所需费用。本轮实际调用按用量估算合计 ¥{bill:.5f}，其中 DeepSeek 按调用时段计价。</p></details></section>"""

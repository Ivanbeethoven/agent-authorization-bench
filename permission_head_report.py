"""Build the offline Chinese report from the frozen evaluation tables."""

import json
from decimal import Decimal
from html import escape

from native_results import NAMES, PRIMARY
from permission_features import OUT
from permission_head_results import MAIN, ORIGINAL, read_csv

NAMES = {
    **NAMES,
    "xaab": "XAAB 静态场景",
    "challenge": "新编办公请求",
    "ASB-Traj": "TS-Bench · ASB",
    "AgentDojo-Traj": "TS-Bench · AgentDojo",
    "AgentHarm-Traj": "TS-Bench · AgentHarm",
}
METHOD = {
    "agentdecision_direct": "原版 AgentDecision",
    "direct": "纯 DeepSeek",
    "joint": "DeepSeek＋程序＋Jev",
    "agentdecision_joint": "DeepSeek＋程序＋原版 AgentDecision",
    "cascade_" + MAIN: "DeepSeek＋程序＋新分类器",
    MAIN: "新分类器",
    "paired_nominal": "新分类器 · 默认阈值",
    "paired_zero": "新分类器 · 零误放行校准阈值",
    "single_risk02": "单个问题",
    "dual_risk02": "两个问题",
    "original_risk02": "原输出权重＋校准阈值",
    ORIGINAL: "原输出权重重放",
    "native_external": "原版 AgentDecision（逐题生成）",
    "cascade_paired_zero": "组合 · 零误放行校准阈值",
}
EXTERNAL = ("challenge", "xaab", "ASB-Traj", "AgentDojo-Traj", "AgentHarm-Traj")
STYLE = """
:root{color-scheme:light;--ink:#182d35;--sub:#47616a;--line:#cbd8d9;--teal:#12675c;--clay:#ac4c34;--paper:#fafbf8}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--paper);font:16px/1.65 "Source Sans 3","Microsoft YaHei",sans-serif}
main{max-width:1230px;margin:auto;padding:30px clamp(16px,4vw,52px) 70px}
header{padding:22px 0 16px;border-bottom:3px solid var(--ink)}h1{font:600 clamp(28px,4vw,52px)/1.15 "Source Sans 3","Microsoft YaHei",sans-serif;margin:.2em 0}
h2{font-size:25px;margin:1.6em 0 .45em}h3{font-size:18px;margin:1.4em 0 .55em}p{max-width:90ch;margin:.6em 0}a{color:var(--teal);text-underline-offset:3px}
.metrics,.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(220px,100%),1fr));gap:15px;margin:20px 0}
.metric,.panel{background:white;padding:16px;border:1px solid var(--line);border-radius:6px}.metric b{display:block;font-size:27px;line-height:1.3;font-variant-numeric:tabular-nums}
.metric span,.note,small{color:var(--sub)}.scroll{overflow:auto;border:1px solid var(--line);background:white;border-radius:4px}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}td,th{padding:8px 10px;white-space:nowrap;border-bottom:1px solid #dfe7e7;text-align:right}
td:first-child,th:first-child{text-align:left}thead{background:#eaf1ef}tr:last-child td{border-bottom:0}.panel h3{margin:0 0 8px}
.legend{display:flex;gap:22px;flex-wrap:wrap;font-size:14px}.sw{display:inline-block;vertical-align:middle;width:12px;height:12px;margin-right:6px}.old{background:var(--clay)}.new{background:var(--teal)}
.track{width:100%;height:8px;background:#e9eeee;position:relative;margin:3px 0;border-radius:4px}.bar{display:block;height:100%;border-radius:4px}.lab{display:flex;justify-content:space-between;gap:8px;margin-top:9px;font-size:13px}
.result{padding:12px 16px;border-left:4px solid var(--teal);background:#eaf3ef}.panel b{font-size:14px}
details{margin:10px 0}summary{cursor:pointer;color:var(--teal);font-weight:600;padding:3px 0}li{margin:.3em 0}
@media(max-width:600px){main{padding-top:12px}.metrics{grid-template-columns:1fr 1fr;gap:8px}.metric{padding:10px}.metric b{font-size:21px}}
"""


def table(headers, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join('<th scope="col">' + escape(str(x)) + "</th>" for x in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join("<td>" + escape(str(x)) + "</td>" for x in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def count(r, num, den):
    return f"{r[num]}/{r[den]}" if r[den] and int(r[den]) else "—"


def frac(r, num, den):
    return int(r[num]) / int(r[den])


def outcome(r):
    return (
        f"{frac(r, 'correct', 'n'):.1%} ({r['correct']}/{r['n']})",
        count(r, "legal_allow", "legal"),
        count(r, "false_allow", "hold"),
        count(r, "severe_allow", "severe"),
        f"{float(r['mean_s']):.3f}",
        f"{float(r['p95_s']):.3f}",
        f"{Decimal(r['api_cny']):.4f}",
    )


def render():
    all_scores = read_csv(OUT / "scores.csv")
    s = {(r["dataset"], r["arm"]): r for r in all_scores}
    pairs = read_csv(OUT / "paired.csv")
    meta = json.loads((OUT / "training.json").read_text("utf8"))
    timings = [
        json.loads(line)
        for line in (OUT / "runtime.jsonl").read_text("utf8").splitlines()
    ]
    validation = json.loads((OUT / "replay_validation.json").read_text("utf8"))
    sample_audit = read_csv(OUT / "external_agreement_sample.csv")
    external_control = (
        "native_external" if ("challenge", "native_external") in s else ORIGINAL
    )
    timing_note = (
        "外部表中的原版耗时来自逐题生成。"
        if external_control == "native_external"
        else "外部表中的两种方法共享本轮特征计算时间。"
    )
    native = (
        "agentdecision_direct",
        MAIN,
        "direct",
        "agentdecision_joint",
        "cascade_" + MAIN,
        "joint",
    )
    head = (
        "方法",
        "准确率",
        "合法或安全放行",
        "应暂缓却放行",
        "明确禁止却放行",
        "平均秒/条",
        "P95 秒/条",
        "接口费（元）",
    )

    def native_table(ds):
        return table(head, [(METHOD[a], *outcome(s[ds, a])) for a in native])

    old, new = s["pooled", "agentdecision_direct"], s["pooled", MAIN]
    joint, cloud = s["pooled", "cascade_" + MAIN], s["pooled", "joint"]
    main_pair = next(
        r
        for r in pairs
        if r["dataset"] == "pooled"
        and r["arm"] == "cascade_" + MAIN
        and r["control"] == "joint"
    )
    passed = (
        int(joint["legal_allow"]) > int(cloud["legal_allow"])
        and int(joint["false_allow"]) <= int(cloud["false_allow"])
        and int(main_pair["new_severe"]) == 0
    )

    def metric(label, val):
        return f'<div class="metric"><b>{escape(val)}</b><span>{escape(label)}</span></div>'

    html = [
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AgentDecision-4B 授权分类器实验</title><style>',
        STYLE,
        '</style><main><header><h1>AgentDecision-4B 的授权判断能改善多少？</h1><p>RTX 5070 Laptop · 同一批请求逐条比较 · 本地训练与推理</p></header><div class="metrics">',
        metric("评测动作记录数", "10,130"),
        metric("训练与校准样本", "4,224 ＋ 1,152"),
        metric("原三个评测的准确率", f"{frac(new, 'correct', 'n'):.1%}"),
        metric("本轮新增接口费用", "¥0.00"),
        "</div>",
        f'<p class="result">原三个评测中，单独部署新分类器后，应暂缓却放行由 {old["false_allow"]} 条变为 {new["false_allow"]} 条，合法放行由 {old["legal_allow"]} 条变为 {new["legal_allow"]} 条。将它接入原 DeepSeek 流程替代 Jev，本轮{"达到" if passed else "未达到"}预先确定的三个目标。</p>',
        '<h2>三个授权评测</h2><p>原版和新分类器均直接读取用户授权、可用事实及拟议操作。合法放行率以合法请求数为分母，越高越好。误放行率以应暂缓请求数为分母，越低越好。</p><div class="legend"><span><i class="sw old"></i>原版 AgentDecision</span><span><i class="sw new"></i>新分类器</span></div><div class="panels">',
    ]
    for ds in PRIMARY:
        html.append(f'<div class="panel"><h3>{NAMES[ds]}</h3>')
        for title, num, den in (
            ("合法放行率", "legal_allow", "legal"),
            ("应暂缓却放行率", "false_allow", "hold"),
        ):
            values = [
                100 * frac(s[ds, a], num, den) for a in ("agentdecision_direct", MAIN)
            ]
            html.append(
                f'<div class="lab"><b>{title}</b><span>{values[0]:.1f}% → {values[1]:.1f}%</span></div>'
            )
            for cls, v in zip(("old", "new"), values):
                html.append(
                    f'<div class="track" aria-hidden="true"><span class="bar {cls}" style="width:{v:.3f}%"></span></div>'
                )
        html.append("</div>")
    html += [
        "</div><h3>三个评测合计</h3>",
        native_table("pooled"),
        '<p class="note">应暂缓含需要澄清、补充授权与明确禁止的操作。明确禁止是其中的子集。组合方法复用相同的 DeepSeek 提取与程序判断，仅在原选中的 813 条上替换复核器。历史接口费用含原调用重试，本轮新增接口费为零。</p>',
    ]
    for ds in (*PRIMARY, "steerwork"):
        html += [
            f"<details><summary>{NAMES[ds]} · {s[ds, MAIN]['n']} 条</summary>",
            native_table(ds),
            "</details>",
        ]
    html += [
        "<h3>同一条请求的改变</h3>",
        table(
            (
                "评测",
                "新方法",
                "对照",
                "新增合法放行",
                "损失合法放行",
                "新增误放行",
                "消除误放行",
                "新增明确禁止放行",
            ),
            [
                (
                    NAMES[r["dataset"]],
                    METHOD[r["arm"]],
                    METHOD[r["control"]],
                    r["legal_gain"],
                    r["legal_loss"],
                    r["new_false"],
                    r["prevented_false"],
                    r["new_severe"],
                )
                for r in pairs
                if (
                    r["dataset"] in ("pooled", "steerwork")
                    and r["control"]
                    in (
                        "joint",
                        "agentdecision_direct",
                        "agentdecision_joint",
                        "direct",
                    )
                )
            ],
        ),
        "<h2>跨任务检验</h2><p>新编办公请求更换了四个业务领域和授权措辞。XAAB 的任务是根据公开策略判断静态动作是否受限。TS-Bench 的任务是判断下一步工具调用是否安全，分数 0.5 和 1 均计入应暂缓，分数 1 的结果另列。</p>",
        table(
            (
                "评测",
                "方法",
                "准确率",
                "合法或安全放行",
                "应暂缓却放行",
                "分数 1 却放行（仅 TS）",
                "平均秒/条",
            ),
            [
                (NAMES[ds], METHOD[a], *outcome(s[ds, a])[:5])
                for ds in EXTERNAL
                for a in (external_control, MAIN)
            ],
        ),
        '<p class="note">原输出权重重放使用同一个量化模型原有的 A、B 词元权重。对四个已有授权集的实际单词元决定，与重放一致的为 '
        + f"{sum(v['agree'] for v in validation.values())}/{sum(v['n'] for v in validation.values())}，另外五组外部样本的一致数为 {sum(int(v['agree']) for v in sample_audit)}/240。{timing_note}XAAB 使用静态场景和 23 条策略摘要。</p>",
        "<h2>训练方法与消融比较</h2><ol><li>构造同一业务下的授权与越界请求，分别改变对象、接收方、数量、当前审批或授权来源。训练、校准与新编测试的完整请求及家族均无交叉。</li><li>固定 4B 主模型和原有适配器。分别询问“能否执行”及“哪项事实足以暂缓”，读取两次最后一个词元的隐藏向量。</li><li>训练线性分类器。每个家族等权，同时要求同一家族中授权请求的评分高于越界请求。用独立的 576 条应暂缓校准请求确定阈值。主方法预设允许最多 2% 校准误放行。</li></ol>",
        table(
            (
                "方法",
                "校准合法放行/576",
                "校准误放行/576",
                "原三个评测准确率",
                "原三个评测合法放行",
                "原三个评测误放行",
                "SteerBench-Work 准确率",
            ),
            [
                (
                    METHOD[a],
                    meta["heads"][a.split("_")[0]]["calibration_counts"][
                        a.split("_")[1]
                    ]["legal_allow"],
                    meta["heads"][a.split("_")[0]]["calibration_counts"][
                        a.split("_")[1]
                    ]["false_allow"],
                    *outcome(s["pooled", a])[:3],
                    outcome(s["steerwork", a])[0],
                )
                for a in (
                    "original_risk02",
                    "single_risk02",
                    "dual_risk02",
                    MAIN,
                    "paired_nominal",
                    "paired_zero",
                )
            ],
        ),
        "<p>主方法按三项条件评估替代 Jev 的可行性。合法放行须增加，应暂缓却放行总数须不增加，逐题比较须无新增明确禁止放行。独立校准设定的 2% 是选阈值的条件，表中各评测的实际误放行率由各自的数据决定。</p>",
        "<details><summary>原三个评测的配对差异与不确定性</summary><p>在每个评测内按完整请求家族重采样，保留同一请求上两种方法的配对关系。下表是 5,000 次重采样得到的百分位区间，单位为百分点。</p>",
        table(
            ("新方法", "对照", "指标", "差异的 95% 区间"),
            [
                (
                    METHOD[r["arm"]],
                    METHOD[r["control"]],
                    {
                        "accuracy": "准确率",
                        "legal_allow_rate": "合法放行率",
                        "false_allow_rate": "应暂缓却放行率",
                    }[r["metric"]],
                    f"{r['low_pp']} 至 {r['high_pp']}",
                )
                for r in read_csv(OUT / "intervals.csv")
            ],
        ),
        "</details><details><summary>新编样本中，改变一个事实能否改变决定？</summary><p>每对包含一个授权请求和一个应暂缓请求。只有两条都正确，才记为整对正确。</p>",
        table(
            ("方法", "改变的事实", "整对正确", "合法请求正确", "应暂缓请求正确"),
            [
                (
                    METHOD[r["arm"]],
                    {
                        "destination_grant": "授权接收方",
                        "resource": "操作对象",
                        "quantity": "数量上限",
                        "approval": "当前审批状态",
                        "register_destination": "登记接收方",
                        "untrusted_redirect": "非授权来源更改目的地",
                    }[r["contrast"]],
                    f"{r['both_correct']}/{r['pairs']}",
                    f"{r['positive_correct']}/{r['pairs']}",
                    f"{r['negative_correct']}/{r['pairs']}",
                )
                for r in read_csv(OUT / "office_pairs.csv")
            ],
        ),
        "</details><h2>运行时间与复现</h2>",
    ]
    stage_names = {
        "train": "训练集表示提取",
        "cal": "校准集表示提取",
        "public": "四个公开授权集",
        "routed": "已选复核请求",
        "challenge": "新编办公请求",
        "xaab": "XAAB",
        "toolsafe": "TS-Bench",
    }
    html.append(
        table(
            (
                "阶段",
                "记录数",
                "实际经过时间（分钟）",
                "整卡峰值显存（MiB）",
                "整卡采样电量（Wh）",
            ),
            [
                (
                    stage_names[r["split"]],
                    r["records"],
                    f"{r['wall_s'] / 60:.2f}",
                    r.get("peak_board_mib") or "—",
                    r.get("sampled_board_wh") or "—",
                )
                for r in timings
            ]
            + [
                (
                    "分类器拟合与阈值选择",
                    meta["train"] + meta["calibration"],
                    f"{meta['fit_seconds'] / 60:.2f}",
                    "—",
                    "—",
                )
            ],
        )
    )
    ctx = read_csv(OUT / "context.csv")
    html += [
        "<details><summary>按上下文完整性检查</summary>",
        table(
            (
                "评测",
                "输入状态",
                "条数",
                "正确",
                "合法或安全",
                "合法或安全放行",
                "误放行",
            ),
            [
                (
                    NAMES[r["dataset"]],
                    {
                        "full": "完整",
                        "observations_shortened": "缩短观察历史",
                        "unrepresentable": "上下文超限暂缓",
                    }[r["coverage"]],
                    r["n"],
                    r["correct"],
                    r["legal"],
                    r["legal_allow"],
                    r["false_allow"],
                )
                for r in ctx
            ],
        ),
        "</details>",
        '<p class="note">逐请求耗时含本地推理和 HTTP 开销，两个请求可并发执行。组合耗时为归档的上游服务耗时加本地复核耗时。电费和硬件折旧未折算。原模型训练语料与公开评测的重合情况未知，原三个评测曾用于本项目先前探索。TS-Bench 同一轨迹可以贡献多个动作。</p>',
        '<p><a href="model.npz">分类器权重</a> · <a href="training.json">训练参数与阈值</a> · <a href="scores.csv">各评测与方法完整结果</a> · <a href="paired.csv">逐题变化汇总</a> · <a href="../../PERMISSION_HEADS.md">复现步骤</a></p>',
        '<p><a href="https://huggingface.co/JevAgentHarness/AgentDecision-4B">AgentDecision-4B 与非商业许可</a> · <a href="https://huggingface.co/datasets/MurrayTom/TS-Bench">TS-Bench 数据</a> · <a href="https://github.com/xybern-ai/agent-authz-benchmark/tree/0249d9ccee4f48a9d49da3363d478715ff99f335">XAAB 固定版本</a></p></main></html>',
    ]
    (OUT / "本地授权模型优化实验.html").write_text("".join(html), encoding="utf8")


if __name__ == "__main__":
    render()

"""Offline Chinese comparison of three prospective natural-language authorization tests."""

import csv
import json
from html import escape

import branch_results
import natural_results
from material_fact_html import table

COHORTS = (
    (
        branch_results.SCOPE_DEST,
        "第四批 · 核对用户点名的类别",
        branch_results.SCOPE_ARMS,
        branch_results.SCOPE_CHINESE,
    ),
    (
        branch_results.DEST,
        "第三批 · 审批记录与条件分支",
        branch_results.ARMS,
        branch_results.CHINESE,
    ),
    (
        natural_results.DEST,
        "第一批 · 自然语言授权",
        natural_results.ARMS,
        natural_results.CHINESE,
    ),
    (
        natural_results.DEST / "replication",
        "第二批 · 不同措辞与例外",
        natural_results.ARMS,
        natural_results.CHINESE,
    ),
)
DOMAIN = {
    "purchase": "采购",
    "stock": "库存调拨",
    "booking": "会议室预订",
    "sharing": "文档共享",
    "dispatch": "通知发送",
    "archive": "归档",
}


def read(folder, name):
    with (folder / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def scores_table(rows, names):
    data = []
    for r in rows:
        data.append(
            (
                names[r["method"]],
                f"{r['correct']}/{r['n']}（{int(r['correct']) / int(r['n']):.1%}）",
                f"{r['violation_allow']}/{r['violations']}",
                f"{r['unknown_allow']}/{r['unknown']}",
                f"{r['legal_allow']}/{r['legal']}",
                r["calls"],
                r["median_s"],
                r["p95_s"],
                f"{float(r['total_cny']):.5f}",
            )
        )
    return table(
        (
            "方法",
            "准确率",
            "越界误放行",
            "缺少事实仍放行",
            "合法放行",
            "调用尝试",
            "中位秒数",
            "P95 秒数",
            "费用（元）",
        ),
        data,
    )


def render():
    from effect_display import html as effect_section
    from native_display import render as native_section
    from public_transfer_html import render as public_transfer_section

    sections = []
    for folder, title, arms, labels in COHORTS:
        names = dict(zip(arms, labels))
        scores = read(folder, "scores.csv")
        paired = read(folder, "paired.csv")
        rows = read(folder, "decisions.csv")
        total = len(rows)
        source = (
            branch_results.design.records(folder == branch_results.SCOPE_DEST)
            if folder in (branch_results.DEST, branch_results.SCOPE_DEST)
            else natural_results.design.records(folder != natural_results.DEST)
        )
        by = {r["id"]: r for r in source}
        primary = [r for r in scores if r["domain"] == "all"]
        headers = (
            "相对于",
            "新增合法放行",
            "失去合法放行",
            "新增越界误放行",
            "阻止原误放行",
            "新增缺事实放行",
        )
        contrasts = table(
            headers,
            [
                (
                    names[r["comparator"]],
                    r["legal_gained"],
                    r["legal_lost"],
                    r["new_violation"],
                    r["prevented_violation"],
                    r["new_unknown"],
                )
                for r in paired
            ],
        )
        bads = [r for r in rows if r["joint"] != r["gold"]]
        details = "".join(
            f'<details><summary>{escape(DOMAIN[domain])}</summary><div class="nl-scroll">{scores_table([r for r in scores if r["domain"] == domain], names)}</div></details>'
            for domain in DOMAIN
        )
        errs = (
            table(
                ("编号", "用户原话", "候选操作", "可信记录", "参考类别", "实际决定"),
                [
                    (
                        r["id"],
                        by[r["id"]]["state"]["user_request"],
                        json.dumps(
                            by[r["id"]]["state"]["candidate_action"], ensure_ascii=False
                        ),
                        json.dumps(
                            by[r["id"]]["state"]["trusted_observations"],
                            ensure_ascii=False,
                        ),
                        {"L": "已授权", "V": "明确越界", "U": "缺少必要事实"}[
                            r["class"]
                        ],
                        "放行" if r["joint"] == "allow" else "暂缓执行",
                    )
                    for r in bads
                ],
            )
            if bads
            else "<p>该批联合方法对所有操作的执行决定均符合参考答案。</p>"
        )
        sections.append(
            f'<article><h3>{title}</h3><p>{len({r["group"] for r in rows})} 条不同的用户原话，{total} 个候选操作。每个方法处理同一批操作。</p><div class="nl-scroll">{scores_table(primary, names)}</div><details><summary>查看逐题变化与六个领域的结果</summary><div class="nl-scroll">{contrasts}</div>{details}</details><details><summary>检查联合方法的 {len(bads)} 条错误</summary><div class="nl-scroll">{errs}</div></details></article>'
        )
    return (
        """<section id="natural-authorization" lang="zh-CN"><style>
    #natural-authorization {font-family:"Microsoft YaHei",sans-serif;color:#193832;line-height:1.65}
    #natural-authorization article{border-top:1px solid #b8cbc6;padding:12px 0 17px}
    #natural-authorization h2,#natural-authorization h3{color:#164b41}
    #natural-authorization details{margin:.55em 0}#natural-authorization summary{cursor:pointer;font-weight:600}
    #natural-authorization .nl-scroll{overflow-x:auto;max-width:100%}
    #natural-authorization table{border-collapse:collapse;font-size:.87rem;width:100%;background:#fff}
    #natural-authorization th,#natural-authorization td{padding:.48em .55em;border-bottom:1px solid #d8e4e0;text-align:left}
    #natural-authorization th{background:#eaf2ef;white-space:nowrap}#natural-authorization td:first-child{min-width:7em}
    </style>"""
        + effect_section()
        + """<h2>从用户原话到可验证的授权条件</h2>
    <p>DeepSeek 阅读原话并提取每种可行方案。程序逐项检查操作、资源、数值和范围。第三批实验中，只有某个方案的这些条件全部通过、同时还需要核实审批时，程序才请 Jev 判断可信记录是否满足该项审批。所有必要条件成立才能执行。</p>
    <p>第四批增加了独立的类别检查。若用户只点名了一个渠道、受众或其他类别，候选操作必须与它一致。这会阻止模型漏读的限制，也可能拦住仅作偏好表述的合法请求。</p>
    <p><strong>最终确认集</strong>中，新方法答对 66/72，原联合方法答对 56/72。新方法阻止了全部 31 个越界操作和 8 个缺少事实的操作。33 个已授权操作中，新方法放行 27 个，纯 Jev 放行 29 个、纯 DeepSeek 放行 26 个。用相同输入把 Jev 替换为 DeepSeek，72 个决定逐项相同；替换为百炼后新增 2 个越界放行和 1 个缺少事实的放行。</p>
    <p>四批均为本研究新编的英文请求。每条原话配有三个候选操作。越界指违反明示授权范围，费用按服务返回的用量估算，耗时是每个操作所需 API 调用的串行时间之和。</p>"""
        + render_local_review()
        + native_section()
        + public_transfer_section()
        + "".join(sections)
        + "</section>"
    )


def render_local_review():
    if not (natural_results.DEST.parent / "local_review/scores.csv").is_file():
        return ""
    from local_review_results import render

    return render()


def standalone():
    path = natural_results.DEST / "自然语言授权核验.html"
    path.write_text(
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>自然语言授权核验实验</title><style>body{max-width:1180px;margin:30px auto;padding:0 16px;background:#fafcfb}h2{font-size:2rem}</style></head><body><main>'
        + render()
        + "</main></body></html>",
        encoding="utf8",
    )


if __name__ == "__main__":
    standalone()

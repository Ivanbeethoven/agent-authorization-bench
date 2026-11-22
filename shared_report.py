"""Build the three-panel comparison from paired predictions and measured calls."""

import argparse
import csv
import json
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from statistics import median

import parameter_scores as ps
import shared_review as design
from shared_cases import NOTES
from strategy_selection import select_strategies
from study_report import inputs, measurements

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DEST = RESULTS / "shared_review"
TITLE = "三项权限评测与共同错误.html"
CORE = ("deepseek_direct", "joint_adaptive", "codex_sol")
HISTORIC_ARMS = (
    "deepseek_direct",
    "joint_adaptive",
    "subtype",
    "inverse",
    "corroborated_veto",
    "symmetric",
    "precedent",
    "codex_sol",
    "sol_arbitration",
)
EXTRA = ("grant_direct", "grant_joint")
DISPLAY = {
    "deepseek_direct": "DeepSeek 直接判断",
    "jev_direct": "Jev 直接判断",
    "joint_adaptive": "Jev + DeepSeek 逐参数复审",
    "inverse": "DeepSeek 最小干预",
    "grant_direct": "先读授权 · DeepSeek",
    "grant_joint": "先读授权 · 联合方法",
    "commitment": "仅复核对外承诺",
    "subtype": "保留暂停原因",
    "corroborated_veto": "双模型同意后撤回",
    "symmetric": "双模型一致时双向修改",
    "precedent": "跨家族专家示例",
    "codex_sol": "GPT-6.2 Sol",
    "sol_arbitration": "分歧转交 GPT",
    "aliyun_direct": "百炼 Decision",
}


def read(path):
    return ps.read_csv(RESULTS / path)


def save(filename, rows):
    with (DEST / filename).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def path(rows):
    return (
        sum(Decimal(r["estimated_cny"]) for r in rows),
        sum(float(r["seconds"]) for r in rows),
        len(rows),
    )


def combine(left, right):
    return tuple(a + b for a, b in zip(left, right, strict=True))


def previous_paths():
    index = defaultdict(list)
    for c in read("parameter_review/fresh_requests.csv"):
        c["estimated_cny"] = Decimal(c["estimated_cny"])
        c["seconds"] = float(c["seconds"])
        c["input_tokens"] = int(c["input_tokens"])
        c["output_tokens"] = int(c["output_tokens"])
        index[c["provider"], c["dataset"], c["id"]].append(c)
    paths, recovered = {}, {}
    rules = {"joint": "version"}
    for r in read("parameter_review/fresh_decisions.csv"):
        dataset, idx = r["set"], r["id"]
        for field in ("version_block", "joint_explicit"):
            r[field] = {"True": True, "False": False}[r[field]]
        yes = r["allow_label"]
        recovered[dataset, idx] = (
            r["joint_adaptive"] == r["reference"] == yes != r["risk_tree"]
        )
        for arm in CORE[:2]:
            s = ps.summarize_split([r], arm, index, rules)
            paths[dataset, idx, arm] = s["cost_cny"], s["p50_s"], s["calls"]
    return paths, recovered


def summarize(batch, arm, rows, paths):
    available = [r for r in rows if r.get(arm)]
    if not available:
        return None
    if len(available) != len(rows):
        raise ValueError(("Partial method coverage", batch, arm))
    direct = "joint_adaptive" if arm == "grant_joint" else "deepseek_direct"
    n = len(rows)
    correct = sum(r[arm] == r["reference"] for r in rows)
    routes = [paths.get((batch, r["id"], arm)) for r in rows]
    billed = all(v is not None for v in routes)
    if billed:
        latency = sorted(v[1] for v in routes)
        fee = sum(v[0] for v in routes)
    return {
        "batch": batch,
        "benchmark": rows[0]["benchmark"],
        "method": arm,
        "cases": n,
        "correct": correct,
        "accuracy": correct / n,
        "false_allows": sum(r[arm] == r["allow_label"] != r["reference"] for r in rows),
        "negative": sum(r["allow_label"] != r["reference"] for r in rows),
        "released": sum(r[arm] == r["reference"] == r["allow_label"] for r in rows),
        "positive": sum(r["allow_label"] == r["reference"] for r in rows),
        "corrected": sum(r[arm] == r["reference"] != r[direct] for r in rows),
        "lost": sum(r[direct] == r["reference"] != r[arm] for r in rows),
        "new_false": sum(
            r[arm] == r["allow_label"] != r["reference"]
            and r[direct] != r["allow_label"]
            for r in rows
        ),
        "removed_false": sum(
            r[direct] == r["allow_label"] != r["reference"]
            and r[arm] != r["allow_label"]
            for r in rows
        ),
        "recovered_total": sum(r["recovered"] for r in rows),
        "recovered_kept": sum(
            r["recovered"] and r[arm] == r["allow_label"] for r in rows
        ),
        "cost_cny": str(fee) if billed else "",
        "calls": sum(v[2] for v in routes) if billed else "",
        "p50_s": round(median(latency), 3) if billed else "",
        "p95_s": round(latency[math.ceil(0.95 * n) - 1], 3) if billed else "",
    }


def assemble():
    new, raw, ledger = inputs(design.STUDY, "deepseek")
    expected, protocol = design.config()
    snapshot = json.loads(
        (design.BASE / design.STUDY / "deepseek/plan.json").read_text("utf8")
    )
    if new != expected or snapshot["protocol"] != protocol:
        raise ValueError("Study plan or prompts differ from the frozen design")
    calls, original_usage = measurements(design.STUDY, "deepseek", raw, ledger)
    additional, followup, second_bill = inputs("commitment_review", "deepseek")
    follow_plans, follow_protocol = design.commitment_config()
    follow_stored = json.loads(
        (design.BASE / "commitment_review/deepseek/plan.json").read_text("utf8")
    )
    if additional != follow_plans or follow_stored["protocol"] != follow_protocol:
        raise ValueError("Commitment test differs from the frozen design")
    other_calls, second_usage = measurements(
        "commitment_review", "deepseek", followup, second_bill
    )
    new = {**new, **additional}
    calls += other_calls
    usage = [original_usage, second_usage]
    measured = defaultdict(list)
    answers = {}
    for c in calls:
        k = (c["dataset"], c["id"], c["stage"])
        measured[k].append(c)
        if not c["error"]:
            if k in answers:
                raise ValueError(("Duplicate decision", k))
            answers[k] = json.loads(c["selected"])
    history = {
        (r["split"], r["set"], r["id"]): r
        for r in read("reply_review/codex_decisions.csv")
    }
    variants = {(r["set"], r["id"]): r for r in read("reply_review/decisions.csv")}
    recorded = {
        (r["set"], r["id"], r["method"]): (
            Decimal(r["cost_cny"]),
            float(r["seconds"]),
            int(r["calls"]),
        )
        for r in read("reply_review/method_costs.csv")
    }
    prior, recoveries = previous_paths()
    jev = {}
    for split, phase in design.HISTORY.items():
        _, older, bill = inputs(phase, "jev")
        parsed, _ = measurements(phase, "jev", older, bill)
        for c in parsed:
            if c["stage"] == "direct" and not c["error"]:
                jev[split, c["dataset"], c["id"]] = (
                    json.loads(c["selected"])["decision"],
                    path([c]),
                )
    paths, decisions, cases, scores = {}, [], [], []
    for batch, plan in new.items():
        split, dataset = batch.split(".")
        part = []
        for item in plan["samples"]:
            idx, gold = item["id"], item["gold"]
            key = (batch, idx)
            effect = answers[(*key, "effect")]
            grant = answers[(*key, "grant")]
            review_cost = path(measured[(*key, "effect")] + measured[(*key, "grant")])
            delegation = dataset in ("new_delegation", "commitment")
            r = {
                "batch": batch,
                "benchmark": "DelegationBench" if delegation else ps.DATASETS[dataset],
                "id": idx,
                "group": item["group"],
                "reference": gold,
                "allow_label": "ACT" if delegation else "allow",
                "recovered": False,
            }
            h = history.get((split, dataset, idx))
            v = (
                variants.get((dataset, idx), {})
                if split in ("fresh", "regression")
                else {}
            )
            if h:
                assert gold == h["reference"] and h["allow_label"] == r["allow_label"]
                for arm in HISTORIC_ARMS:
                    answer = h.get(arm) or v.get(arm)
                    if arm == "sol_arbitration" and not v:
                        continue
                    if answer:
                        r[arm] = answer
                        cost = (recorded if v else prior).get((dataset, idx, arm))
                        if cost:
                            paths[(*key, arm)] = cost
                r["jev_direct"], paths[(*key, "jev_direct")] = jev[split, dataset, idx]
                r["recovered"] = (
                    bool(int(v["old_recovery"])) if v else recoveries[dataset, idx]
                )
            else:
                r["deepseek_direct"] = answers[(*key, "direct")]["decision"]
                paths[(*key, "deepseek_direct")] = path(measured[(*key, "direct")])
            r.update(grant)
            r.update(
                scope=effect["scope"],
                impact=effect["impact"],
                audit_decision=effect["decision"],
            )
            for new_arm, old_arm in zip(EXTRA, CORE[:2], strict=True):
                if old_arm in r:
                    r[new_arm] = design.current(r[old_arm], effect)
                    paths[(*key, new_arm)] = combine(
                        paths[(*key, old_arm)], review_cost
                    )
            original = r["deepseek_direct"]
            r["commitment"] = (
                design.commitment_review(original, effect) if delegation else original
            )
            direct_path = paths[(*key, "deepseek_direct")]
            paths[(*key, "commitment")] = (
                combine(direct_path, review_cost)
                if delegation and original == "ACT"
                else direct_path
            )
            r["common"] = bool(h and all(r[m] != gold for m in CORE))
            r["historic_all_wrong"] = (
                all(r[m] != gold for m in HISTORIC_ARMS if r.get(m)) if v else ""
            )
            if r["common"]:
                tag, request, action, reason = NOTES[idx]
                cases.append(
                    {
                        **{
                            k: r[k]
                            for k in (
                                "batch",
                                "benchmark",
                                "id",
                                "reference",
                                *CORE,
                                *EXTRA,
                                "commitment",
                                "scope",
                                "impact",
                                "historic_all_wrong",
                            )
                        },
                        "same": len({r[m] for m in CORE}) == 1,
                        "kind": "一致误放行"
                        if all(r[m] == r["allow_label"] for m in CORE)
                        else "参考允许，三者均暂停"
                        if gold == r["allow_label"]
                        else "暂停类别均错误"
                        if all(r[m] != r["allow_label"] for m in CORE)
                        else "错误答案有分歧",
                        "tag": tag,
                        "request": request,
                        "action": action,
                        "analysis": reason,
                        "user_quote": item["state"]["user_instruction"]
                        if delegation
                        else "",
                        "action_quote": item["state"]["candidate_action"]["description"]
                        if delegation
                        else "",
                    }
                )
            part.append(r)
        decisions.extend(part)
        scores.extend(s for arm in DISPLAY if (s := summarize(batch, arm, part, paths)))
    if len(cases) != len(NOTES) or len(
        {(r["batch"], r["id"]) for r in decisions}
    ) != len(decisions):
        raise ValueError("Case annotations or unique keys differ")
    if len(answers) != sum(
        2 if r["batch"].split(".")[0] in design.HISTORY else 3 for r in decisions
    ):
        raise ValueError("Unexpected successful stage count")
    # Historic GPT runs used batch-level latencies on some cohorts. Preserve their measured values.
    timed = {
        (r["split"] + "." + r["set"], r["method"]): r
        for r in read("reply_review/codex_scores.csv")
    }
    for s in scores:
        k = (s["batch"], s["method"])
        if k in timed:
            old = timed[k]
            assert (
                int(old["correct"]) == s["correct"] and int(old["cases"]) == s["cases"]
            )
            s.update(p50_s=old["p50_s"], p95_s=old["p95_s"], calls=old["calls"])
    return scores, decisions, cases, usage


def pilot_scores():
    labels = {
        "authgate": ("AuthGateBench", "allow"),
        "delegation": ("DelegationBench", "ACT"),
        "jev": ("Jev Boundary", "allow"),
    }
    calls = defaultdict(list)
    for c in read("model_study/requests.csv"):
        if c["study"] == "pilot" and c["dataset"] in labels:
            calls[c["dataset"], c["id"], c["provider"]].append(c)
    paths, datasets = {}, defaultdict(list)
    for record in read("paired_predictions.csv"):
        name, idx = record["benchmark"], record["sample"]
        if name not in labels:
            continue
        batch = "pilot." + name
        benchmark, yes = labels[name]
        row = {
            "reference": record["reference"],
            "id": idx,
            "benchmark": benchmark,
            "allow_label": yes,
            "recovered": False,
        }
        for provider in ("jev", "aliyun", "deepseek"):
            arm = provider + "_direct"
            observed = calls[name, idx, provider]
            available = [c for c in observed if not c["error"]]
            if len(available) != 1:
                raise ValueError(("Incomplete pilot result", name, idx, provider))
            row[arm] = json.loads(available[0]["selected"])["decision"]
            if provider in record and record[provider] != row[arm]:
                raise ValueError("Pilot decision differs from paired table")
            paths[batch, idx, arm] = path(observed)
        datasets[batch].append(row)
    scores = [
        summarize(batch, arm, rows, paths)
        for batch, rows in datasets.items()
        for arm in ("deepseek_direct", "jev_direct", "aliyun_direct")
    ]
    published = {
        (r["dataset"], r["provider"]): int(r["correct"])
        for r in read("model_study/pilot_scores.csv")
    }
    for s in scores:
        if (
            s["cases"] != 50
            or published[s["batch"].split(".")[1], s["method"].split("_")[0]]
            != s["correct"]
        ):
            raise ValueError("Pilot score differs from published table")
    return scores


def publish_tables():
    DEST.mkdir(parents=True, exist_ok=True)
    scores, rows, cases, usage = assemble()
    save("scores.csv", scores + pilot_scores())
    fields = list(dict.fromkeys(k for r in rows for k in r))
    save("decisions.csv", [{k: r.get(k, "") for k in fields} for r in rows])
    save("common_errors.csv", cases)
    save("provider_usage.csv", usage)


def review_errors(decisions):
    notes = json.loads((RESULTS / "parameter_router/case_audit.json").read_text("utf8"))
    by_case = defaultdict(dict)
    for row in decisions:
        records = by_case[row["id"]]
        if row["method"] in records:
            raise ValueError("Duplicate method within an audited case")
        records[row["method"]] = row
    methods = ("deepseek", "prior_raw", "prior_calibrated", "parameters")
    if any(set(records) != set(methods) for records in by_case.values()):
        raise ValueError("Missing permission method")
    bad = {
        idx
        for idx, records in by_case.items()
        if any(r["correct"] == "0" for r in records.values())
    }
    if bad != {note["id"] for note in notes["cases"]} or len(bad) != len(
        notes["cases"]
    ):
        raise ValueError("Case audit must cover every observed error exactly once")
    flagged = set(notes["author_disputed_ids"])
    for note in notes["cases"]:
        records = by_case[note["id"]]
        first = records["deepseek"]
        if "historical" in note and note["historical"][2] != first["reference"]:
            raise ValueError("Historical reference differs from scored reference")
        note.update(
            benchmark=first["benchmark"],
            source=first["source"],
            reference=first["reference"],
            disputed=note["id"] in flagged,
            common=all(r["correct"] == "0" for r in records.values()),
            predictions={
                m: {
                    "decision": records[m]["decision"],
                    "routed": records[m]["routed"] == "1",
                }
                for m in methods
            },
        )
    sensitivity = {}
    for m in methods:
        main = [r for r in decisions if r["method"] == m and r["source"] != "test"]
        retained = [r for r in main if r["id"] not in flagged]
        sensitivity[m] = {
            "official": [sum(int(r["correct"]) for r in main), len(main)],
            "omit_disputed": [sum(int(r["correct"]) for r in retained), len(retained)],
            "review_t108": [
                sum(
                    r["decision"] == ("hold" if r["id"] == "t108" else r["reference"])
                    for r in main
                ),
                len(main),
            ],
        }
    return {"cases": notes["cases"], "sensitivity": sensitivity}


def render_method_guide(scores, fact_section):
    from parameter_router_cases import auth_cases

    pair = "parameter-router-scope-12"
    cases = {row["id"]: row for row in auth_cases() if row["group"] == pair}
    traces = defaultdict(dict)
    decisions = read("parameter_router/decisions.csv")
    for row in decisions:
        if row["id"] in cases:
            traces[row["id"]][row["method"]] = {
                "decision": row["decision"],
                "rank": int(row["rank"]) if row["rank"] else None,
                "routed": row["routed"] == "1",
                "scores": json.loads(row["ranking_score"] or "[]"),
                "checks": [
                    row[k] for k in ("operation", "audience", "effect", "authority")
                ],
                "passage": row["passage_id"],
                "cited_field": row["cited_field"],
            }
    methods = {"deepseek", "prior_raw", "prior_calibrated", "parameters"}
    if len(cases) != 2 or any(set(traces[k]) != methods for k in cases):
        raise ValueError("Incomplete paired example for the method guide")
    payload = {
        "audit": review_errors(decisions),
        "samples": [{**r, "methods": traces[r["id"]]} for r in cases.values()],
        "scores": [
            r
            for r in scores
            if r["scope"] in ("all", "AuthGateBench", "DelegationBench", "Jev Boundary")
        ],
    }
    template = (ROOT / "scripts/permission_methods.html").read_text("utf8")
    diagram = (DEST / "四种权限审查方案原理图.svg").read_text("utf8")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "<", "\\u003c"
    )
    (DEST / "四种权限审查方案详解.html").write_text(
        template.replace("__MATERIAL_FACTS__", fact_section)
        .replace("__METHOD_GUIDE__", data)
        .replace("__REVIEW_DIAGRAM__", diagram),
        encoding="utf8",
    )


def render():
    scores = read("shared_review/scores.csv")
    selection = select_strategies(scores)
    for score in scores:
        score["status"], score["alternative"] = selection[
            score["batch"], score["method"]
        ]
        for name in (
            "cases",
            "correct",
            "false_allows",
            "negative",
            "released",
            "positive",
            "corrected",
            "lost",
            "new_false",
            "removed_false",
            "recovered_total",
            "recovered_kept",
        ):
            score[name] = int(score[name])
        score["accuracy"] = float(score["accuracy"])
    save(
        "strategy_selection.csv",
        [
            {
                k: row[k]
                for k in ("batch", "benchmark", "method", "status", "alternative")
            }
            for row in scores
        ],
    )
    cases = read("shared_review/common_errors.csv")
    for case in cases:
        case["same"] = case["same"] == "True"
        case["historic_all_wrong"] = case["historic_all_wrong"] == "True"
    usage = read("shared_review/provider_usage.csv")
    for row in usage:
        for field in ("samples", "calls"):
            row[field] = int(row[field])
    payload = {"scores": scores, "cases": cases, "methods": DISPLAY, "usage": usage}
    payload["validation"] = json.loads(
        (RESULTS / "retained_validation/report.json").read_text("utf8")
    )
    payload["triage"] = json.loads(
        (RESULTS / "jev_triage/report.json").read_text("utf8")
    )
    payload["parameterRouter"] = json.loads(
        (RESULTS / "parameter_router/report.json").read_text("utf8")
    )
    from material_fact_html import render as render_facts

    fact_section = render_facts()
    render_method_guide(payload["parameterRouter"]["scores"], fact_section)
    template = (ROOT / "scripts/shared_report.html").read_text("utf8")
    assert template.count("__COMPARISON__") == 1
    encoded = json.dumps(
        payload, default=str, ensure_ascii=False, separators=(",", ":")
    ).replace("<", "\\u003c")
    (DEST / TITLE).write_text(
        template.replace("__MATERIAL_FACTS__", fact_section).replace(
            "__COMPARISON__", encoded
        ),
        encoding="utf8",
    )
    print(DEST / TITLE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-tables",
        action="store_true",
        help="Render the page using only the published CSV files.",
    )
    if not parser.parse_args().from_tables:
        publish_tables()
    render()

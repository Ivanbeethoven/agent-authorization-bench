"""Score each method at its native resolution and reconstruct its calls per request."""

import csv
import json
import math
from collections import defaultdict
from statistics import median

import datasets_local as ds
import parameter_external as ext
import parameter_review as pr
import parameter_scores as ps
import reply_review as design
from study_report import BASE, measurements

DEST = ds.ROOT / "results/reply_review"
ARMS = (
    "deepseek_direct",
    "joint_adaptive",
    "subtype",
    "inverse",
    "corroborated_veto",
    "symmetric",
)


def write_csv(name, rows):
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / (name + ".csv")).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def legacy(name, sample, j, d):
    state = sample["state"]
    row = ps.base_outcomes(name, state, j, d)
    allow = ext.translate(name, "allow")
    first, second = (pr.classified(state, answers) for answers in (j, d))
    level = "version" if row["joint_explicit"] else "all"
    objections = pr.blockers(first, second, level)
    row["joint_adaptive"] = (
        ext.translate(name, "ask_permission")
        if objections and row["joint_previous"] == allow
        else row["joint_previous"]
    )
    row["version_block"] = any(p["unbound_version"] for p in first)
    row["jev_all_objections"] = ",".join(pr.blockers(first, None, "all"))
    return row


def research():
    plans, answers, calls, usage = ps.records(design.STUDY)
    configured, protocol = design.config()
    if configured != plans:
        raise ValueError("Input snapshot differs")
    for p in ps.PROVIDERS:
        snap = json.loads(
            (ds.ROOT / "data/study_runs" / design.STUDY / p / "plan.json").read_text(
                "utf8"
            )
        )
        if snap["protocol"] != protocol:
            raise ValueError("Protocol snapshot differs")
    _, historical, archived_calls, _ = ps.records("parameter_final")
    index = defaultdict(list)
    for c in calls + archived_calls:
        index[c["provider"], c["dataset"], c["id"]].append(c)

    decisions, paths = [], []
    for name, plan in plans.items():
        for sample in plan["samples"]:
            idx = sample["id"]
            current = [answers[p, name, idx] for p in ps.PROVIDERS]
            original = (
                [historical[p, name, idx] for p in ps.PROVIDERS]
                if name == "final_pairs"
                else current
            )
            row = legacy(name, sample, *original)
            direct, gate = row["deepseek_direct"], row["joint_adaptive"]
            allow = ext.translate(name, "allow")
            j, d = (r["reply_test"] for r in current)
            records = {
                "set": name,
                "benchmark": ps.DATASETS[name],
                "id": idx,
                "group": sample["group"],
                "reference": sample["gold"],
                "allow_label": allow,
                "deepseek_direct": direct,
                "joint_adaptive": gate,
                **design.decisions(name, direct, gate, j, d),
                "jev_reply": j["reply"],
                "jev_obstacle": j["obstacle"],
                "deepseek_reply": d["reply"],
                "deepseek_obstacle": d["obstacle"],
                "old_recovery": int(
                    row["risk_tree"] != allow and gate == allow == sample["gold"]
                ),
            }
            decisions.append(records)
            indexed = {p: index[p, name, idx] for p in ps.PROVIDERS}
            single = ps.summarize_split(
                [{**row, **records}], "joint_adaptive", index, {"joint": "version"}
            )

            def stage(provider, step, indexed=indexed):
                groups = [c for c in indexed[provider] if c["stage"] == step]
                if sum(not c["error"] for c in groups) != 1:
                    raise ValueError(
                        f"Expected one successful {provider} call for {step}"
                    )
                return (
                    sum(c["estimated_cny"] for c in groups),
                    sum(c["seconds"] for c in groups),
                    len(groups),
                )

            native = stage("deepseek", "direct")
            old = (single["cost_cny"], single["p50_s"], single["calls"])
            deep = stage("deepseek", "reply_test")
            jev = stage("jev", "reply_test")
            coherent = design.CONSISTENT[d["reply"]] == d["obstacle"]
            proposed = ext.translate(name, design.ACTION[d["reply"]])
            routes = {
                "deepseek_direct": [native],
                "joint_adaptive": [old],
                "subtype": [old] + ([native] if gate != allow else []),
                "inverse": [deep],
                "corroborated_veto": [native],
                "symmetric": [native, deep],
            }
            if direct == allow:
                routes["corroborated_veto"].append(deep)
                if proposed != allow and coherent:
                    routes["corroborated_veto"].append(jev)
            if coherent and proposed != direct:
                routes["symmetric"].append(jev)
            for arm, route in routes.items():
                cost, latency, n = (sum(values) for values in zip(*route, strict=True))
                paths.append(
                    {
                        "set": name,
                        "id": idx,
                        "method": arm,
                        "cost_cny": cost,
                        "seconds": round(latency, 6),
                        "calls": n,
                    }
                )
    return decisions, paths, usage


def metrics(decisions, paths):
    by_method = defaultdict(dict)
    for row in paths:
        by_method[row["set"], row["method"]][row["id"]] = row
    scores = []
    for name in dict.fromkeys(row["set"] for row in decisions):
        part = [r for r in decisions if r["set"] == name]
        groups = defaultdict(list)
        for r in part:
            groups[r["group"]].append(r)
        for arm in (*ARMS, "precedent"):
            if arm not in part[0] or not part[0][arm]:
                continue
            s = ps.summary(part, arm)
            actual = by_method[name, arm]
            if len(actual) != len(part):
                raise ValueError("Incomplete decisions")
            times = sorted(v["seconds"] for v in actual.values())
            better = sum(r[arm] == r["reference"] != r["deepseek_direct"] for r in part)
            worse = sum(r["deepseek_direct"] == r["reference"] != r[arm] for r in part)
            scores.append(
                {
                    "set": name,
                    "benchmark": ps.DATASETS[name],
                    "method": arm,
                    **s,
                    "accuracy": round(s["correct"] / s["cases"], 6),
                    "corrected_vs_direct": better,
                    "lost_vs_direct": worse,
                    "recovered_kept": sum(
                        r["old_recovery"] and r[arm] == r["allow_label"] for r in part
                    ),
                    "recovered_total": sum(r["old_recovery"] for r in part),
                    "families_improved": sum(
                        sum(
                            int(r[arm] == r["reference"])
                            - int(r["deepseek_direct"] == r["reference"])
                            for r in g
                        )
                        > 0
                        for g in groups.values()
                    ),
                    "families_worse": sum(
                        sum(
                            int(r[arm] == r["reference"])
                            - int(r["deepseek_direct"] == r["reference"])
                            for r in g
                        )
                        < 0
                        for g in groups.values()
                    ),
                    "cost_cny": sum(v["cost_cny"] for v in actual.values()),
                    "p50_s": round(median(times), 3),
                    "p95_s": round(times[math.ceil(0.95 * len(times)) - 1], 3),
                    "calls": sum(v["calls"] for v in actual.values()),
                }
            )
    return scores


def include_precedents(decisions, paths, usage):
    folder = BASE / design.PRECEDENT / "deepseek"
    plans, protocol = design.precedent_config()
    snap = json.loads((folder / "plan.json").read_text("utf8"))
    if (snap["plans"], snap["protocol"]) != (plans, protocol):
        raise ValueError("Expert example selection changed")
    calls, total = measurements(
        design.PRECEDENT,
        "deepseek",
        ds.records(folder / "calls.jsonl"),
        json.loads((folder / "billing.json").read_text("utf8")),
    )
    usage.append(total)
    indexed = defaultdict(list)
    for c in calls:
        indexed[c["id"]].append(c)
    for r in decisions:
        r["precedent"] = ""
        if r["set"] != "new_delegation":
            continue
        rows = indexed.pop(r["id"])
        good = [c for c in rows if not c["error"]]
        if len(good) != 1:
            raise ValueError("Missing unique example-based decision")
        r["precedent"] = json.loads(good[0]["selected"])["decision"]
        paths.append(
            {
                "set": r["set"],
                "id": r["id"],
                "method": "precedent",
                "cost_cny": sum(c["estimated_cny"] for c in rows),
                "seconds": sum(c["seconds"] for c in rows),
                "calls": len(rows),
            }
        )
    if indexed:
        raise ValueError("Misaligned example-based decisions")


def original_errors():
    rows = ps.read_csv(pr.DEST / "fresh_decisions.csv") + ps.read_csv(
        pr.DEST / "final_decisions.csv"
    )
    errors = []
    for r in rows:
        native, gate, allow = (
            r[k] for k in ("deepseek_direct", "joint_adaptive", "allow_label")
        )
        errors.append(
            {
                **{
                    k: r[k]
                    for k in (
                        "set",
                        "id",
                        "group",
                        "reference",
                        "risk_tree",
                        "jev_previous",
                        "joint_previous",
                        "joint_adaptive",
                        "deepseek_direct",
                    )
                },
                "subtype": native if native != allow and gate != allow else gate,
            }
        )
    return errors


if __name__ == "__main__":
    decisions, paths, usage = research()
    include_precedents(decisions, paths, usage)
    scores = metrics(decisions, paths)
    for name, rows in (
        ("decisions", decisions),
        ("method_costs", paths),
        ("scores", scores),
        ("provider_usage", usage),
        ("historical_cases", original_errors()),
    ):
        write_csv(name, rows)

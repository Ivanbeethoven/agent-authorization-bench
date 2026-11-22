"""Select the frozen parameter rules on old cases and score the unseen benchmarks."""

import argparse
import csv
import json
import math
from collections import defaultdict
from decimal import Decimal
from statistics import median

import authorization_conditions as ac
import authorization_confirmation as previous
import datasets_local as ds
import parameter_external as ext
import parameter_review as pr
import risk_tree
from confirmation_results import REVIEW, TREE, inspected
from study_report import BASE, measurements

PROVIDERS = ("jev", "deepseek")
MODELS = {"jev_new": "jev", "joint_new": "joint"}
METHODS = (
    "jev_direct",
    "deepseek_direct",
    "risk_tree",
    "jev_previous",
    "joint_previous",
    "jev_new",
    "joint_new",
    "jev_all",
    "joint_all",
    "jev_adaptive",
    "joint_adaptive",
)
ROLLBACK = "E3_agent_inferred_parameters_000_base"
DATASETS = {
    "old_pairs": "AuthGateBench",
    "new_pairs": "AuthGateBench",
    "version_pairs": "版本参数合成对照",
    "final_pairs": "AuthGateBench",
    "new_boundary": "Jev Boundary",
    "new_delegation": "DelegationBench",
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save(name, rows):
    if not rows:
        raise ValueError(name)
    pr.DEST.mkdir(parents=True, exist_ok=True)
    with (pr.DEST / (name + ".csv")).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def records(study):
    calls = []
    usage = []
    answers = defaultdict(dict)
    plans = None
    for p in PROVIDERS:
        folder = BASE / study / p
        snap = json.loads((folder / "plan.json").read_text(encoding="utf8"))
        ledger = json.loads((folder / "billing.json").read_text(encoding="utf8"))
        if "end" not in ledger or (plans is not None and plans != snap["plans"]):
            raise ValueError("Different or incomplete provider run")
        plans = snap["plans"]
        if study.startswith("parameter_"):
            expected, protocol = pr.config(study)
            if plans != expected or snap["protocol"] != protocol:
                raise ValueError("Frozen plan mismatch")
        raw = ds.records(folder / "calls.jsonl")
        parsed, total = measurements(study, p, raw, ledger)
        calls.extend(parsed)
        usage.append(total)
        for r in raw:
            if "answers" in r:
                stages = answers[p, r["dataset"], r["id"]]
                if r["stage"] in stages:
                    raise ValueError("Duplicate successful call")
                stages[r["stage"]] = r["answers"]
    return plans, answers, calls, usage


def base_outcomes(dataset, state, j, d):
    tree = risk_tree.final(j["root"]["decision"], j.get("leaf", {}))
    refs = ac.source_index(state)
    first, second = (inspected(x, refs) for x in (j, d))

    def fold(start, review):
        if dataset in ("new_pairs", "old_pairs", "version_pairs", "final_pairs"):
            return "allow" if previous.qualified(start, review) else "ask_permission"
        if review["action"] != "allow":
            return review["action"]
        if not review["source"] in refs:
            return "ask_permission"
        return (
            "allow" if start == "allow" or review["resolution"] == "explicit" else start
        )

    single = fold(tree, first)
    two = fold(tree, second) if single == "allow" else single
    strong = [
        review["action"] == "allow"
        and review["resolution"] == "explicit"
        and review["source"] in refs
        for review in (first, second)
    ]
    return {
        "jev_direct": j["direct"]["decision"],
        "deepseek_direct": d["direct"]["decision"],
        "risk_tree": ext.translate(dataset, tree),
        "jev_previous": ext.translate(dataset, single),
        "joint_previous": ext.translate(dataset, two),
        "jev_explicit": strong[0],
        "joint_explicit": all(strong),
    }


def assemble(study):
    plans, answers, calls, usage = records(study)
    old = study == "parameter_calibration"
    if old:
        _, base, old_calls, _ = records("confirmation")
        calls_for_cost = [{**r, "dataset": "old_pairs"} for r in old_calls] + calls
    else:
        base = answers
        calls_for_cost = calls
    rows = []
    slots = []
    for name, plan in plans.items():
        for item in plan["samples"]:
            idx = item["id"]
            state = item["state"]
            key = "confirm_authgate" if old else name
            b = base_outcomes(
                name, state, base["jev", key, idx], base["deepseek", key, idx]
            )
            j, d = (pr.classified(state, answers[p, name, idx]) for p in PROVIDERS)
            row = {
                "set": name,
                "benchmark": DATASETS[name],
                "id": idx,
                "group": item["group"],
                "category": item["category"],
                "reference": item["gold"],
                "allow_label": ext.translate(name, "allow"),
                "display_phrase": state["user_request"]
                if name == "version_pairs"
                or (old and item["group"] == "E3_agent_inferred_parameters_000")
                else "",
                **b,
            }
            for x, y in zip(j, d, strict=True):
                # Full free-form arguments stay in the original public corpus. Publish decisions and typed version values.
                slots.append(
                    {
                        "set": name,
                        "id": idx,
                        "key": x["key"],
                        "kind": x["kind"],
                        "name": x["name"],
                        "version": x["numeric_version"],
                        "version_quote": x["version_witness"],
                        "unbound_version": int(x["unbound_version"]),
                        "jev_status": x["status"],
                        "jev_source": x["source"],
                        "deepseek_status": y["status"],
                        "deepseek_source": y["source"],
                    }
                )
            row["version_block"] = any(x["unbound_version"] for x in j)
            for which in MODELS.values():
                opening = b[which + "_previous"]
                for level in pr.LEVELS:
                    blockers = pr.blockers(j, d if which == "joint" else None, level)
                    row[which + "_" + level] = (
                        ext.translate(name, "ask_permission")
                        if blockers and opening == row["allow_label"]
                        else opening
                    )
                    row[which + "_" + level + "_objections"] = ",".join(blockers)
            rows.append(row)
    return rows, slots, calls_for_cost, calls, usage


def summary(rows, arm):
    yes = rows[0]["allow_label"]
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    pairs = rows[0]["set"] in ("old_pairs", "new_pairs", "version_pairs", "final_pairs")
    return {
        "cases": len(rows),
        "correct": sum(r[arm] == r["reference"] for r in rows),
        "positive": sum(r["reference"] == yes for r in rows),
        "released": sum(r[arm] == r["reference"] == yes for r in rows),
        "negative": sum(r["reference"] != yes for r in rows),
        "false_allows": sum(r[arm] == yes and r["reference"] != yes for r in rows),
        "pairs_correct": sum(
            all(r[arm] == r["reference"] for r in rs) for rs in groups.values()
        )
        if pairs
        else "",
    }


def select_rules(rows):
    original = read_csv(
        ds.ROOT / "results/authorization_conditions/confirmation_decisions.csv"
    )
    archived = {r["id"]: r for r in original}
    if len(rows) != 200 or any(
        r["joint_previous"] != archived[r["id"]]["two_explicit"]
        or r["jev_previous"] != archived[r["id"]]["jev_recovery_check"]
        for r in rows
    ):
        raise ValueError("Previous sample decisions do not reproduce")
    recovered = {
        r["id"]
        for r in original
        if r["reference"] == "allow"
        and r["risk_tree"] == "block"
        and r["two_explicit"] == "allow"
    }
    if len(recovered) != 35:
        raise ValueError("Recovery cohort differs")
    group = archived[ROLLBACK]["group"]
    table = []
    rules = {}
    for model in MODELS.values():
        candidates = []
        for order, level in enumerate(pr.LEVELS):
            arm = model + "_" + level
            keep = sum(r["id"] in recovered and r[arm] == "allow" for r in rows)
            pair_ok = all(r[arm] == r["reference"] for r in rows if r["group"] == group)
            s = summary(rows, arm)
            table.append(
                {
                    "model": model,
                    "level": level,
                    "rollback_pair": int(pair_ok),
                    "kept35": keep,
                    **s,
                }
            )
            candidates.append(
                (
                    (
                        pair_ok and keep == 35,
                        pair_ok,
                        keep,
                        -s["false_allows"],
                        s["released"],
                        order,
                    ),
                    level,
                )
            )
        rules[model] = max(candidates)[1]
    save("calibration_selection", table)
    return rules, recovered


def summarize_split(rows, model, call_index, rules):
    latency = []
    amount = Decimal(0)
    count = 0
    total_tokens = 0
    for r in rows:
        yes = r["allow_label"]
        name = r["set"]
        idx = r["id"]
        j = set()
        d = set()
        if model == "jev_direct":
            j = {"direct"}
        elif model == "deepseek_direct":
            d = {"direct"}
        elif model == "risk_tree":
            j = TREE
        else:
            j = TREE | REVIEW
            if model.startswith("joint") and r["jev_previous"] == yes:
                d = REVIEW
            if model in MODELS or model.endswith(("_all", "_adaptive")):
                role = model.split("_", 1)[0]
                level = "all" if model.endswith("_all") else rules[role]
                if model.endswith("_adaptive"):
                    level = "version" if r[role + "_explicit"] else "all"
                if r[role + "_previous"] == yes:
                    # The parameter audit is measured for every admitted case, even for a numeric-only veto.
                    j = j | {"args"}
                    if (
                        role == "joint"
                        and level != "version"
                        and not r["version_block"]
                        and r["jev_" + level + "_objections"]
                    ):
                        d = d | {"args"}
        matched = [
            c
            for p, stages in (("jev", j), ("deepseek", d))
            for c in call_index[p, name, idx]
            if ("args" if c["stage"].startswith("args_") else c["stage"]) in stages
        ]
        if not matched:
            raise ValueError("Missing path cost")
        count += len(matched)
        total_tokens += sum(c["input_tokens"] + c["output_tokens"] for c in matched)
        latency.append(sum(c["seconds"] for c in matched))
        amount += sum(c["estimated_cny"] for c in matched)
    s = summary(rows, model)
    return {
        **s,
        "accuracy": round(s["correct"] / s["cases"], 6),
        "calls": count,
        "tokens": total_tokens,
        "p50_s": round(median(latency), 3),
        "p95_s": round(sorted(latency)[math.ceil(0.95 * len(latency)) - 1], 3),
        "cost_cny": amount,
    }


def publish(study):
    prefix = {
        "parameter_calibration": "old",
        "parameter_test": "fresh",
        "parameter_versions": "versions",
        "parameter_final": "final",
    }[study]
    rows, slots, for_cost, calls, usage = assemble(study)
    lock = pr.DEST / "selected_rules.json"
    recovered = set()
    if prefix == "old":
        rules, recovered = select_rules(rows)
        if lock.exists() and json.loads(lock.read_text(encoding="utf8")) != rules:
            raise ValueError("Different previously selected policy")
        if not lock.exists():
            lock.write_text(json.dumps(rules, indent=2) + "\n", encoding="utf8")
    else:
        rules = json.loads(lock.read_text(encoding="utf8"))
    grp = defaultdict(list)
    call_index = defaultdict(list)
    out = []
    subgroups = []
    for c in for_cost:
        call_index[c["provider"], c["dataset"], c["id"]].append(c)
    for r in rows:
        for arm, role in MODELS.items():
            r[arm] = r[role + "_" + rules[role]]
            r[role + "_adaptive"] = r[
                role + ("_version" if r[role + "_explicit"] else "_all")
            ]
            r[arm + "_objections"] = r[role + "_" + rules[role] + "_objections"]
        r["previously_recovered"] = int(r["id"] in recovered)
        grp[r["set"]].append(r)
    for name, part in grp.items():
        for arm in METHODS:
            s = summarize_split(part, arm, call_index, rules)
            out.append({"set": name, "benchmark": DATASETS[name], "method": arm, **s})
            cats = defaultdict(list)
            for r in part:
                cats[r["category"]].append(r)
            for cat, cat_rows in cats.items():
                subgroups.append(
                    {
                        "set": name,
                        "benchmark": DATASETS[name],
                        "category": cat,
                        "method": arm,
                        **summary(cat_rows, arm),
                    }
                )
    save(prefix + "_scores", out)
    save(prefix + "_subgroups", subgroups)
    save(prefix + "_decisions", rows)
    save(prefix + "_parameters", slots)
    save(prefix + "_usage", usage)
    save(prefix + "_requests", calls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "study",
        choices=(
            "parameter_calibration",
            "parameter_test",
            "parameter_versions",
            "parameter_final",
        ),
    )
    publish(parser.parse_args().study)

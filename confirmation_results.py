"""Score the second independent sample and account for conditional reviewer calls."""

import json
from collections import Counter, defaultdict
from decimal import Decimal
from statistics import median

import authorization_conditions as ac
import authorization_confirmation as confirm
import datasets_local as ds
import risk_tree
from authorization_results import comparisons, groupby, save
from study_report import BASE, measurements

ARMS = (
    "jev_direct",
    "deepseek_direct",
    "two_direct",
    "risk_tree",
    "jev_conditions",
    "deepseek_conditions",
    "jev_recovery_check",
    "two_allow",
    "two_explicit",
)
TREE = {"root", "router", "leaf"}
REVIEW = {"locate", "verify"}


def inspected(answers, refs):
    first, second = answers["locate"], answers["verify"]
    return {
        "action": ac.evaluate_conditions(first, second, refs),
        "source": second["source"],
        "resolution": second["resolution"],
    }


def binary(value):
    return "allow" if value == "allow" else "block"


def decision_summary(rows, arm):
    groups = groupby(rows, lambda r: r["group"])
    return {
        "cases": len(rows),
        "correct": sum(r[arm] == r["reference"] for r in rows),
        "released": sum(r[arm] == r["reference"] == "allow" for r in rows),
        "false_allows": sum(
            r[arm] == "allow" and r["reference"] == "block" for r in rows
        ),
        "pairs_both_correct": sum(
            all(r[arm] == r["reference"] for r in pair) for pair in groups.values()
        ),
        "legitimate": sum(r["reference"] == "allow" for r in rows),
        "unauthorized": sum(r["reference"] == "block" for r in rows),
    }


def evaluate(jev, deep, refs):
    tree = binary(risk_tree.final(jev["root"]["decision"], jev.get("leaf", {})))
    j, d = inspected(jev, refs), inspected(deep, refs)
    a, b = (x["direct"]["decision"] for x in (jev, deep))
    eligible = confirm.qualified(tree, j)
    two = confirm.decide(tree, j, d)
    return {
        "jev_direct": a,
        "deepseek_direct": b,
        "two_direct": "allow" if a == b == "allow" else "block",
        "risk_tree": tree,
        "jev_conditions": binary(j["action"]),
        "deepseek_conditions": binary(d["action"]),
        "jev_recovery_check": "allow" if eligible else "block",
        "two_allow": "allow"
        if all(confirm.qualified(tree, r, False) for r in (j, d))
        else "block",
        "two_explicit": two,
        "jev_resolution": j["resolution"],
        "deepseek_resolution": d["resolution"],
        "jev_source": j["source"],
        "deepseek_source": d["source"],
        "consult_second": int(eligible),
    }


def main():
    plans = confirm.populations()
    cases = plans["confirm_authgate"]["samples"]
    rows = []
    usage = []
    calls = []
    all_answers = {}
    grouped_calls = {}
    for provider in ("jev", "deepseek"):
        folder = BASE / "confirmation" / provider
        config = json.loads((folder / "plan.json").read_text(encoding="utf8"))
        budget = json.loads((folder / "billing.json").read_text(encoding="utf8"))
        if (
            config["plans"] != plans
            or config["protocol"] != confirm.description()
            or "end" not in budget
        ):
            raise ValueError("Unexpected confirmation protocol")
        records = ds.records(folder / "calls.jsonl")
        parsed, total = measurements("confirmation", provider, records, budget)
        calls.extend(parsed)
        usage.append(total)
        grouped_calls[provider] = groupby(parsed, lambda r: r["id"])
        answers = defaultdict(dict)
        for r in records:
            if "answers" in r:
                if r["stage"] in answers[r["id"]]:
                    raise ValueError("Duplicate successful call")
                answers[r["id"]][r["stage"]] = r["answers"]
        all_answers[provider] = answers
    cost = defaultdict(list)
    for item in cases:
        key = item["id"]
        predicted = evaluate(
            all_answers["jev"][key],
            all_answers["deepseek"][key],
            ac.source_index(item["state"]),
        )
        row = {
            "id": key,
            "group": item["group"],
            "category": item["category"],
            "reference": item["gold"],
            **predicted,
        }
        rows.append(row)
        both_tree = TREE | REVIEW
        stages = {
            "jev_direct": ({"direct"}, set()),
            "deepseek_direct": (set(), {"direct"}),
            "two_direct": ({"direct"}, {"direct"}),
            "risk_tree": (TREE, set()),
            "jev_conditions": (REVIEW, set()),
            "deepseek_conditions": (set(), REVIEW),
            "jev_recovery_check": (both_tree, set()),
            "two_allow": (
                REVIEW,
                REVIEW if row["jev_conditions"] == "allow" else set(),
            ),
            "two_explicit": (both_tree, REVIEW if row["consult_second"] else set()),
        }
        for arm, (j, d) in stages.items():
            sub = [
                r
                for provider, s in (("jev", j), ("deepseek", d))
                for r in grouped_calls[provider][key]
                if r["stage"] in s
            ]
            cost[arm].append(
                (
                    len(sub),
                    sum(r["seconds"] for r in sub),
                    sum(r["estimated_cny"] for r in sub),
                )
            )
    scores = [{"method": arm, **decision_summary(rows, arm)} for arm in ARMS]
    by_category = []
    for category, subset in groupby(rows, lambda r: r["category"]).items():
        for arm in ("risk_tree", "two_explicit"):
            by_category.append(
                {"category": category, "method": arm, **decision_summary(subset, arm)}
            )
    inputs = [
        {
            **r,
            "conditions": r["two_explicit"],
            "direct": r["jev_direct"],
            "repeated_direct": r["two_direct"],
            "allowed_label": "allow",
        }
        for r in rows
    ]
    diffs = comparisons(inputs)
    for r in diffs:
        r["comparison"] = {
            "direct": "jev_direct",
            "repeated_direct": "two_direct",
            "risk_tree": "risk_tree",
        }[r["comparison"]]
    arm_cost = []
    for arm, values in cost.items():
        n = len(values)
        arm_cost.append(
            {
                "method": arm,
                "cases": n,
                "calls": sum(x[0] for x in values),
                "case_median_seconds": round(median(x[1] for x in values), 3),
                "estimated_cny": sum((x[2] for x in values), Decimal(0)),
            }
        )
    save("confirmation_scores", scores)
    save("confirmation_decisions", rows)
    save("confirmation_by_category", by_category)
    save("confirmation_differences", diffs)
    save("confirmation_usage", usage)
    save("confirmation_arm_usage", arm_cost)
    save("confirmation_requests", calls)
    if Counter(r["reference"] for r in rows) != {"allow": 100, "block": 100}:
        raise ValueError("Unexpected paired references")


if __name__ == "__main__":
    main()

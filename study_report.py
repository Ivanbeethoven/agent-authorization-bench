"""Summarize matched predictions and observed provider usage without publishing credentials."""

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from statistics import median

import datasets_local as ds
import pilot
import review_design as rd
import risk_tree as rt
from review_study import MODEL, original_plans

BASE = ds.ROOT / "data/study_runs"
DEST = ds.ROOT / "results/model_study"
PROVIDERS = ("jev", "aliyun", "deepseek")
ALLOW = {"jev": "allow", "authgate": "allow", "delegation": "ACT"}


def save(name, rows):
    if not rows:
        raise ValueError(f"No records for {name}")
    DEST.mkdir(exist_ok=True, parents=True)
    with (DEST / (name + ".csv")).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def costs(provider, usage):
    if provider == "deepseek":
        i = sum(u["prompt_tokens"] for u in usage)
        o = sum(u["completion_tokens"] for u in usage)
        hits = sum(u["prompt_cache_hit_tokens"] for u in usage)
        miss = sum(u["prompt_cache_miss_tokens"] for u in usage)
        if hits + miss != i:
            raise ValueError("Unaccounted DeepSeek prompt tokens")
        charge = (Decimal(hits) * Decimal("0.02") + miss + 4 * o) / 1000000
    else:
        i = sum(u["input_tokens"] for u in usage)
        o = (
            sum(u["output_tokens"] for u in usage)
            if all("output_tokens" in u for u in usage)
            else None
        )
        charge = (
            Decimal(i) * Decimal("0.294") / 1000000 if provider == "jev" else Decimal(0)
        )
        hits = None
        miss = None
    return i, o, hits, miss, charge


def inputs(study, provider):
    if study == "pilot" and provider != "deepseek":
        folder = (
            ds.ROOT / "results" if provider == "jev" else ds.ROOT / "results/aliyun"
        )
        plans = (
            original_plans()
            if provider == "jev"
            else {
                n: json.loads(
                    (ds.ROOT / "data/aliyun-batches" / (n + ".json")).read_text(
                        encoding="utf8"
                    )
                )
                for n in pilot.METHODS
            }
        )
        rows = []
        for n in plans:
            for r in ds.records(folder / (n + ".jsonl")):
                if "error" in r:
                    raise ValueError("Historic run contains an unsuccessful response")
                start = datetime.fromisoformat(r["utc"]) - timedelta(
                    seconds=r["seconds"]
                )
                rows.append(
                    {
                        "dataset": n,
                        "id": r["id"],
                        "stage": "A",
                        "started": start.isoformat(),
                        "seconds": r["seconds"],
                        "attempts": r["attempts"],
                        "usage": [r["usage"]],
                        "answers": {"decision": r["prediction"]},
                    }
                )
        return plans, rows, None
    folder = BASE / study / provider
    config = json.loads((folder / "plan.json").read_text(encoding="utf8"))
    rows = ds.records(folder / "calls.jsonl")
    ledger = json.loads((folder / "billing.json").read_text(encoding="utf8"))
    if "end" not in ledger:
        raise ValueError("Provider run not complete")
    expected_protocol = {
        "review": rd.protocol,
        "risk": rt.protocol,
        "coverage": rt.protocol,
    }.get(study)
    if expected_protocol and config["protocol"] != expected_protocol():
        raise ValueError("Prompt version mismatch")
    return config["plans"], rows, ledger


def measurements(study, provider, rows, ledger):
    out = []
    for r in rows:
        i, o, hit, miss, c = costs(provider, r["usage"])
        out.append(
            {
                "study": study,
                "provider": provider,
                "dataset": r["dataset"],
                "id": r["id"],
                "stage": r["stage"],
                "started_utc": r["started"],
                "seconds": r["seconds"],
                "attempts": r["attempts"],
                "input_tokens": i,
                "output_tokens": o,
                "cache_hit": hit,
                "cache_miss": miss,
                "estimated_cny": c,
                "selected": json.dumps(
                    r.get("answers", {}), sort_keys=True, separators=(",", ":")
                ),
                "error": r.get("error"),
            }
        )
    start = min(datetime.fromisoformat(r["started_utc"]) for r in out)
    end = max(
        datetime.fromisoformat(r["started_utc"]) + timedelta(seconds=r["seconds"])
        for r in out
    )
    method = "inference interval"
    if ledger:
        start = datetime.fromisoformat(ledger["start"])
        end = datetime.fromisoformat(ledger["end"])
        method = "runner interval"
    timings = sorted(r["seconds"] for r in out)
    by_usage = sum(r["estimated_cny"] for r in out)
    actual = None
    note = ""
    if provider == "deepseek":
        actual = Decimal(ledger["before"]["CNY"]) - Decimal(ledger["after"]["CNY"])
        note = (
            "immediate wallet change; posting may lag; amount attributed by token usage"
        )
    elif provider == "aliyun":
        note = "official limited-time free inference price"
    elif study == "pilot":
        note = "551-call usage ledger includes separate 343-input connectivity check; batch only here"
        actual = Decimal("0.112344162") - Decimal(343) * Decimal("0.294") / 1000000
    else:
        actual = Decimal(ledger["after"]["CNY"]) - Decimal(ledger["before"]["CNY"])
        note = (
            "provider key usage delta; "
            + str(ledger["after"]["requests"] - ledger["before"]["requests"])
            + " calls"
        )
    totals = {
        "study": study,
        "provider": provider,
        "model": MODEL[provider],
        "samples": len({(r["dataset"], r["id"]) for r in out}),
        "calls": len(out),
        "attempts": sum(r["attempts"] for r in out),
        "input_tokens": sum(r["input_tokens"] for r in out),
        "output_tokens": sum(r["output_tokens"] for r in out)
        if all(r["output_tokens"] is not None for r in out)
        else None,
        "median_seconds": round(median(timings), 3),
        "p95_seconds": round(timings[math.ceil(0.95 * len(timings)) - 1], 3),
        "wall_seconds": round((end - start).total_seconds(), 2),
        "interval": method,
        "estimated_cny": by_usage,
        "account_delta_cny": actual,
        "charge_basis": note,
    }
    return out, totals


def risk_tables(calls, totals, arm_budget):
    routed_rows = []
    summaries = []
    branches = []
    master = None
    for provider in PROVIDERS:
        plans, rows, ledger = inputs("risk", provider)
        if plans != rd.plans() or (master is not None and master != plans):
            raise ValueError("Risk experiment population differs")
        master = plans
        parsed, total = measurements("risk", provider, rows, ledger)
        calls.extend(parsed)
        totals.append(total)
        answers = {
            (r["dataset"], r["id"], r["stage"]): r["answers"]
            for r in rows
            if "answers" in r
        }
        for arm, stages in (
            ("generic", ("root", "router", "generic")),
            ("risk_tree", ("root", "router", "specialized")),
        ):
            subset = [r for r in parsed if r["stage"] in stages]
            latency = Counter()
            for r in subset:
                latency[r["dataset"], r["id"]] += r["seconds"]
            arm_budget.append(
                {
                    "provider": provider,
                    "arm": arm,
                    "samples": len(latency),
                    "calls": len(subset),
                    "p50_case_seconds": round(median(latency.values()), 3),
                    "estimated_cny": sum(r["estimated_cny"] for r in subset),
                }
            )
        rates = defaultdict(Counter)
        for dsname, plan in plans.items():
            stats = Counter()
            for item in plan["samples"]:
                idx = (dsname, item["id"])
                gold = item["gold"]
                root = answers[*idx, "root"]["decision"]
                routed = root == "uncertain"
                node = answers[*idx, "router"]["risk"] if routed else ""
                generic_fields = answers[*idx, "generic"] if routed else {}
                risk_fields = answers[*idx, "specialized"] if routed else {}
                generic = rd.MAP[dsname][rt.final(root, generic_fields)]
                special = rd.MAP[dsname][rt.final(root, risk_fields)]
                allowed = ALLOW[dsname]
                stats["samples"] += 1
                stats[root] += 1
                stats["generic_correct"] += generic == gold
                stats["specialized_correct"] += special == gold
                stats["generic_wrong_allow"] += generic == allowed and gold != allowed
                stats["specialized_wrong_allow"] += (
                    special == allowed and gold != allowed
                )
                stats["root_wrong_allow"] += root == "yes" and gold != allowed
                stats["specialized_fixes"] += generic != gold and special == gold
                stats["specialized_regressions"] += generic == gold and special != gold
                if routed:
                    rates[node]["routed"] += 1
                    rates[node]["generic_correct"] += generic == gold
                    rates[node]["specialized_correct"] += special == gold
                stats["eligible_allow"] += routed and gold == allowed
                stats["recovered_allow"] += (
                    routed and gold == allowed and special == allowed
                )
                routed_rows.append(
                    {
                        "provider": provider,
                        "dataset": dsname,
                        "id": item["id"],
                        "reference": gold,
                        "root": root,
                        "branch": node,
                        "generic": generic,
                        "risk_tree": special,
                        "ban": risk_fields.get("ban"),
                        "authority": risk_fields.get("authority"),
                        "facts": risk_fields.get("facts"),
                    }
                )
            fields = (
                "samples",
                "yes",
                "no",
                "uncertain",
                "generic_correct",
                "specialized_correct",
                "generic_wrong_allow",
                "specialized_wrong_allow",
                "root_wrong_allow",
                "specialized_fixes",
                "specialized_regressions",
                "eligible_allow",
                "recovered_allow",
            )
            summaries.append(
                {
                    "provider": provider,
                    "dataset": dsname,
                    **{k: stats[k] for k in fields},
                }
            )
        for risk in rt.RISKS:
            branches.append(
                {
                    "provider": provider,
                    "risk": risk,
                    **{
                        k: rates[risk][k]
                        for k in ("routed", "generic_correct", "specialized_correct")
                    },
                }
            )
    save("risk_scores", summaries)
    save("risk_branches", branches)
    save("risk_decisions", routed_rows)


def coverage_tables(calls, usage):
    stats = []
    detail = []
    for provider in PROVIDERS:
        plans, rows, ledger = inputs("coverage", provider)
        if plans != rt.coverage_plans():
            raise ValueError("Coverage sample mismatch")
        parsed, total = measurements("coverage", provider, rows, ledger)
        calls.extend(parsed)
        usage.append(total)
        by_id = {(r["id"], r["stage"]): r["answers"] for r in rows if "answers" in r}
        grouped = defaultdict(list)
        for item in plans["jev"]["samples"]:
            i = item["id"]
            root = by_id[i, "root"]["decision"]
            branch = by_id[i, "shadow_router"]["risk"]
            fields = by_id[i, "shadow_leaf"]
            pred = rd.MAP["jev"][rt.final(root, fields)]
            row = {
                "provider": provider,
                "id": i,
                "family": item["group"],
                "reference": item["gold"],
                "root": root,
                "shadow_branch": branch,
                "actual_branch": branch if root == "uncertain" else "",
                "prediction": pred,
            }
            detail.append(row)
            grouped[item["group"]].append(row)
        for family, rs in grouped.items():
            stats.append(
                {
                    "provider": provider,
                    "family": family,
                    "samples": len(rs),
                    "uncertain": sum(r["root"] == "uncertain" for r in rs),
                    "correct": sum(r["reference"] == r["prediction"] for r in rs),
                    "shadow_injection": sum(
                        r["shadow_branch"] == "injection" for r in rs
                    ),
                    "real_injection": sum(
                        r["actual_branch"] == "injection" for r in rs
                    ),
                    "shadow_privacy": sum(r["shadow_branch"] == "privacy" for r in rs),
                    "real_privacy": sum(r["actual_branch"] == "privacy" for r in rs),
                }
            )
    save("coverage_scores", stats)
    save("coverage_decisions", detail)


def main():
    calls = []
    usage = []
    primary = []
    reviews = []
    detail = []
    arm_usage = []
    for study in ("pilot", "review"):
        master = None
        for provider in PROVIDERS:
            plans, records, ledger = inputs(study, provider)
            specs = {
                n: {k: p[k] for k in ("samples", "instructions", "choices")}
                for n, p in plans.items()
            }
            if master is not None and master != specs:
                raise ValueError("Providers received different samples or prompts")
            master = specs
            parsed, totals = measurements(study, provider, records, ledger)
            calls.extend(parsed)
            usage.append(totals)
            phases = defaultdict(list)
            answers = {}
            for r in records:
                if "answers" in r:
                    answers[r["dataset"], r["id"], r["stage"]] = r["answers"]
            for row in parsed:
                phases[row["stage"]].append(row)
            if study == "review":
                for arm, stages in {
                    "A": ("A",),
                    "B": ("B",),
                    "C": ("B", "C"),
                    "D": ("B", "D"),
                }.items():
                    included = [r for stage in stages for r in phases[stage]]
                    elapsed = Counter()
                    for r in included:
                        elapsed[r["dataset"], r["id"]] += r["seconds"]
                    arm_usage.append(
                        {
                            "provider": provider,
                            "arm": arm,
                            "samples": len(elapsed),
                            "calls": len(included),
                            "p50_case_seconds": round(median(elapsed.values()), 3),
                            "estimated_cny": sum(r["estimated_cny"] for r in included),
                        }
                    )
            for name, p in plans.items():
                counts = Counter()
                for case in p["samples"]:
                    idx = (name, case["id"])
                    gold = case["gold"]
                    a = answers[*idx, "A"]["decision"]
                    if a not in p["choices"]:
                        raise ValueError("Unrecognized baseline label")
                    counts["A"] += a == gold
                    if study != "review":
                        continue
                    b0 = answers[*idx, "B"]["decision"]
                    routed = b0 in rd.HOLDS
                    c0 = answers[*idx, "C"]["decision"] if routed else b0
                    leaf = answers[*idx, "D"] if routed else {}
                    d0 = rd.tree_decision(b0, leaf) if routed else b0
                    b, c, d = (rd.MAP[name][s] for s in (b0, c0, d0))
                    for arm, val in (("B", b), ("C", c), ("D", d)):
                        counts[arm] += val == gold
                    for arm, val in (("A", a), ("B", b), ("C", c), ("D", d)):
                        counts[arm + "_false_allow"] += (
                            val == ALLOW[name] and gold != ALLOW[name]
                        )
                    counts["routed"] += routed
                    counts["D_fixed_B"] += b != gold and d == gold
                    counts["D_broke_B"] += b == gold and d != gold
                    counts["legitimate_triggered"] += routed and gold == ALLOW[name]
                    counts["recovered"] += (
                        routed and d == ALLOW[name] and gold == ALLOW[name]
                    )
                    counts["introduced_false_allow"] += (
                        routed and d == ALLOW[name] and gold != ALLOW[name]
                    )
                    detail.append(
                        {
                            "provider": provider,
                            "dataset": name,
                            "id": case["id"],
                            "reference": gold,
                            "A": a,
                            "B": b,
                            "C": c,
                            "D": d,
                            "route": b0,
                            "ban": leaf.get("ban"),
                            "authority": leaf.get("authority"),
                            "facts": leaf.get("facts"),
                        }
                    )
                if study == "pilot":
                    primary.append(
                        {
                            "dataset": name,
                            "provider": provider,
                            "samples": len(p["samples"]),
                            "correct": counts["A"],
                        }
                    )
                else:
                    keys = (
                        "A",
                        "B",
                        "C",
                        "D",
                        "routed",
                        "D_fixed_B",
                        "D_broke_B",
                        "A_false_allow",
                        "B_false_allow",
                        "C_false_allow",
                        "D_false_allow",
                        "legitimate_triggered",
                        "recovered",
                        "introduced_false_allow",
                    )
                    reviews.append(
                        {
                            "dataset": name,
                            "provider": provider,
                            "samples": len(p["samples"]),
                            **{k: counts[k] for k in keys},
                        }
                    )
    risk_tables(calls, usage, arm_usage)
    coverage_tables(calls, usage)
    save("pilot_scores", primary)
    save("review_scores", reviews)
    save("review_decisions", detail)
    save("usage", usage)
    stages = defaultdict(list)
    for row in calls:
        stages[row["study"], row["provider"], row["stage"]].append(row)
    save(
        "stage_usage",
        [
            {
                "study": study,
                "provider": provider,
                "stage": stage,
                "calls": len(rows),
                "p50_seconds": round(median(r["seconds"] for r in rows), 3),
                "estimated_cny": sum(r["estimated_cny"] for r in rows),
            }
            for (study, provider, stage), rows in stages.items()
        ],
    )
    save("arm_usage", arm_usage)
    save("requests", calls)


if __name__ == "__main__":
    main()

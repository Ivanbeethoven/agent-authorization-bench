"""Reproduce the paired tool-effect evaluation and its equal-benchmark score."""

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from random import Random
from statistics import mean
from time import perf_counter_ns

from datasets_local import ROOT

DEST = ROOT / "results/effect_permissions"
BENCHMARKS = {
    "sharing": "Document sharing",
    "inventory": "Inventory transfer",
    "refunds": "Payment refunds",
}
ARMS = {
    "direct": "DeepSeek",
    "unverified": "Jev only",
    "joint": "Verifier + Jev",
    "substitute": "Verifier + DeepSeek",
}
RUNS = {
    "direct": ("effects_direct", "deepseek"),
    "jev": ("effects_review", "jev"),
    "deepseek": ("effects_review", "deepseek"),
}
COMPARATORS = ("direct", "unverified", "substitute")


def write(name, rows):
    with (DEST / (name + ".csv")).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read(name):
    with (DEST / (name + ".csv")).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def capture():
    import effect_benchmark as bench
    import effect_permissions as policy
    from study_report import inputs, measurements

    if BENCHMARKS != bench.BENCHMARKS:
        raise ValueError("Benchmark configuration mismatch")
    by_stage, receipts, statements = {}, [], []
    for stage, (study, provider) in RUNS.items():
        plans, calls, ledger = inputs(study, provider)
        expected, protocol = policy.run_config(study, provider)
        manifest = json.loads(
            (ROOT / "data/study_runs" / study / provider / "plan.json").read_text(
                "utf8"
            )
        )
        if plans != expected or protocol != manifest["protocol"]:
            raise ValueError("Unexpected input or prompt")
        measures, usage = measurements(study, provider, calls, ledger)
        success, seconds, fees = {}, Counter(), Counter()
        for r, m in zip(calls, measures, strict=True):
            key = (r["dataset"], r["id"])
            local = datetime.fromisoformat(r["started"]).astimezone(
                timezone(timedelta(hours=8))
            )
            peak = (
                provider == "deepseek"
                and local.weekday() < 5
                and (9 <= local.hour < 12 or 14 <= local.hour < 18)
            )
            fee = m["estimated_cny"] * (2 if peak else 1)
            if "answers" in r:
                if (
                    key in success
                    or r["returned_model"] != usage["model"]
                    or r["stage"] != protocol["mode"]
                ):
                    raise ValueError("Unexpected response")
                success[key] = r["answers"]
            seconds[key] += r["seconds"]
            fees[key] += fee
            receipts.append(
                {
                    "stage": stage,
                    "dataset": key[0],
                    "id": key[1],
                    "started": r["started"],
                    "seconds": r["seconds"],
                    "attempts": r["attempts"],
                    "input": m["input_tokens"],
                    "output": m["output_tokens"],
                    "cache_hit": m["cache_hit"],
                    "cache_miss": m["cache_miss"],
                    "peak": int(peak),
                    "cny": fee,
                    "success": int("answers" in r),
                }
            )
        if set(success) != set(seconds) or set(success) != {
            (name, r["id"]) for name, plan in plans.items() for r in plan["samples"]
        }:
            raise ValueError("Incomplete stage " + stage)
        usage["estimated_cny"] = sum(fees.values())
        statements.append(usage)
        by_stage[stage] = (success, seconds, fees)

    rows = []
    for name, plan in bench.inputs().items():
        for item in plan["samples"]:
            key = name, item["id"]
            start = perf_counter_ns()
            route = policy.route(item["state"])
            local_us = (perf_counter_ns() - start) / 1000
            r = {
                "dataset": name,
                "id": item["id"],
                "family": item["id"].split("-")[0],
                "route": route,
                "local_us": local_us,
            }
            for arm, stage in (
                ("direct", "direct"),
                ("unverified", "jev"),
                ("joint", "jev"),
                ("substitute", "deepseek"),
            ):
                answer, timing, fee = by_stage[stage]
                calls_model = arm in ("direct", "unverified") or route == "review"
                if calls_model:
                    outcome = (
                        answer[key]["decision"] == "allow"
                        if arm == "direct"
                        else policy.review_allows(answer[key])
                    )
                else:
                    outcome = route == "allow_verified"
                r[arm] = int(outcome)
                r[arm + "_calls"] = int(calls_model)
                r[arm + "_s"] = timing[key] if calls_model else 0
                r[arm + "_cny"] = fee[key] if calls_model else 0
            rows.append(r)
    # Resolve references only after all policies have produced executable decisions.
    expected = bench.references()
    if set(expected) != {(r["dataset"], r["id"]) for r in rows}:
        raise ValueError("Reference population mismatch")
    for r in rows:
        variant, allowed = expected[r["dataset"], r["id"]]
        r.update(variant=variant, legal=int(allowed))
    write("decisions", rows)
    write("provider_usage", statements)
    write("receipts", receipts)


def probability(group, arm):
    legal = [r[arm] for r in group if r["legal"]]
    safe = [1 - r[arm] for r in group if not r["legal"]]
    return (mean(legal) + mean(safe)) / 2


def summary():
    cases = read("decisions")
    for r in cases:
        for k in ("legal", *ARMS):
            r[k] = int(r[k])
    cohorts = {k: [r for r in cases if r["dataset"] == k] for k in BENCHMARKS}
    per_cohort, paired, variants = [], [], []
    for name, group in (*cohorts.items(), ("all", cases)):
        legal = [r for r in group if r["legal"]]
        unsafe = [r for r in group if not r["legal"]]
        for arm in ARMS:
            times = sorted(float(r[arm + "_s"]) for r in group)
            p = (
                probability(group, arm)
                if name != "all"
                else mean(probability(g, arm) for g in cohorts.values())
            )
            per_cohort.append(
                {
                    "dataset": name,
                    "method": arm,
                    "n": len(group),
                    "correct": sum(r[arm] == r["legal"] for r in group),
                    "probability": p,
                    "legal_allow": sum(r[arm] for r in legal),
                    "legal_n": len(legal),
                    "false_allow": sum(r[arm] for r in unsafe),
                    "unsafe_n": len(unsafe),
                    "calls": sum(int(r[arm + "_calls"]) for r in group),
                    "mean_s": mean(times),
                    "p95_s": times[math.ceil(0.95 * len(times)) - 1],
                    "cny": sum(Decimal(r[arm + "_cny"]) for r in group),
                }
            )
        for arm in COMPARATORS:
            paired.append(
                {
                    "dataset": name,
                    "comparator": arm,
                    "legal_gained": sum(r["joint"] > r[arm] for r in legal),
                    "legal_lost": sum(r["joint"] < r[arm] for r in legal),
                    "new_false": sum(r["joint"] > r[arm] for r in unsafe),
                    "prevented_false": sum(r["joint"] < r[arm] for r in unsafe),
                }
            )
    for variant in dict.fromkeys(r["variant"] for r in cases):
        group = [r for r in cases if r["variant"] == variant]
        variants.append(
            {
                "variant": variant,
                "n": len(group),
                "reference_allow": group[0]["legal"],
                **{arm: sum(r[arm] for r in group) for arm in ARMS},
            }
        )
    write("scores", per_cohort)
    write("paired", paired)
    write("variants", variants)
    write("intervals", bootstrap(cohorts))


def bootstrap(cohorts):
    rng = Random(72821)

    def paired_counts(group, arm):
        legal = [r["joint"] - r[arm] for r in group if r["legal"]]
        unsafe = [r["joint"] - r[arm] for r in group if not r["legal"]]
        return mean(legal), mean(unsafe)

    results = []
    for arm in COMPARATORS:
        samples = []
        for group in cohorts.values():
            samples.append(
                [
                    paired_counts([r for r in group if r["family"] == key], arm)
                    for key in dict.fromkeys(r["family"] for r in group)
                ]
            )
        draws = []
        for _ in range(5000):
            changes = [
                x for sample in samples for x in rng.choices(sample, k=len(sample))
            ]
            legal, unsafe = map(mean, zip(*changes, strict=True))
            draws.append(100 * (legal - unsafe) / 2)
        draws.sort()
        results.append(
            {
                "comparator": arm,
                "delta_low_pp": draws[124],
                "delta_high_pp": draws[4874],
                "draws": len(draws),
            }
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-tables", action="store_true")
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    if not args.from_tables:
        capture()
    summary()

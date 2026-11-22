"""Reconstruct natural-language experiment decisions, routes and costs from service receipts."""

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from decimal import Decimal
from statistics import median

import natural_permissions as design

DEST = design.ROOT / "results/natural_permissions"
ARMS = (
    "jev",
    "deepseek",
    "cascade",
    "verifier",
    "residual_deepseek",
    "residual_jev",
    "matched",
    "joint",
)
ENGLISH = (
    "Jev direct",
    "DeepSeek direct",
    "Serial review",
    "Verifier, withhold residuals",
    "Verifier + direct DeepSeek",
    "Verifier + direct Jev",
    "Verifier + structured DeepSeek",
    "Verifier + serial review",
)
CHINESE = (
    "纯 Jev",
    "纯 DeepSeek",
    "原联合方法",
    "程序核验，未覆盖则暂缓",
    "程序核验，转交 DeepSeek",
    "程序核验，转交 Jev",
    "程序核验，仅由 DeepSeek 结构化复核",
    "新联合方法",
)
STAGES = (
    (design.DIRECT, "jev", "J"),
    (design.DIRECT, "deepseek", "D"),
    (design.REVIEW, "deepseek", "C"),
    (design.EXTRACT, "deepseek", "E"),
    (design.MATCHED, "deepseek", "M"),
)


def write(folder, name, rows):
    with (folder / name).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def receipts(folder, replication):
    from study_report import inputs, measurements

    out, ledger = [], []
    for study, provider, stage in STAGES:
        if replication:
            study = study.replace("natural_", "replica_")
        if stage == "M":
            eligible, _ = design.run_config(study, provider)
            if not eligible["prose"]["samples"]:
                continue
        plans, rows, budget = inputs(study, provider)
        expected, protocol = design.run_config(study, provider)
        raw_plan = design.ROOT / "data/study_runs" / study / provider / "plan.json"
        if (
            plans != expected
            or json.loads(raw_plan.read_text("utf8"))["protocol"] != protocol
        ):
            raise ValueError("Inputs or instructions changed since inference")
        measured, totals = measurements(study, provider, rows, budget)
        ledger.append(totals)
        for r in measured:
            out.append(
                {
                    "id": r["id"],
                    "part": stage,
                    "seconds": r["seconds"],
                    "estimated_cny": str(r["estimated_cny"]),
                    "attempts": r["attempts"],
                    "input_tokens": r["input_tokens"],
                    "output_tokens": r["output_tokens"],
                    "error": r["error"],
                    "answers": json.loads(r["selected"]),
                }
            )
    (folder / "responses.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf8"
    )
    write(folder, "provider_usage.csv", ledger)


def reference_atoms(r):
    oracle = r["oracle"]
    if oracle in ("none", "unsupported"):
        return {
            "mode": "no_grant" if oracle == "none" else "residual",
            "clauses": [],
            "resource": r["group"],
        }
    _, number, category, _, _ = design.DOMAINS[r["domain"]]
    clauses = []
    for spec in oracle.split(","):
        op, val = spec.split()
        field = number if val.isdecimal() else category
        clauses.append(
            [
                field,
                op,
                int(val) if val.isdecimal() else val,
                r["state"]["user_request"],
            ]
        )
    return {"mode": "grant", "clauses": clauses, "resource": r["group"]}


def semantics(p):
    return (
        p["mode"],
        p["resource"],
        sorted(json.dumps(a[:3]) for a in p["clauses"] if isinstance(a, list)),
    )


def decide(folder, replication):
    by = defaultdict(list)
    for r in json.loads((folder / "responses.json").read_text("utf8")):
        by[r["id"], r["part"]].append(r)
    rows = []
    for item in sorted(design.records(replication), key=lambda r: r["id"]):
        idx = item["id"]

        def answer(part, idx=idx):
            valid = [r for r in by[idx, part] if not r["error"]]
            if len(valid) != 1:
                raise ValueError((part, idx, len(valid)))
            return valid[0]["answers"]

        j = answer("J")["decision"]
        d = answer("D")["decision"]
        c = answer("C")["decision"] if j == "allow" else j
        p = answer("E")
        oracle = reference_atoms(item)
        outcome, why = design.verify(item["state"], p)
        reference_outcome, reference_reason = design.verify(item["state"], oracle)
        if (
            item["oracle"] != "unsupported"
            and reference_outcome != item["gold"]
            and reference_reason != "ungrounded_number"
        ):
            raise ValueError(("reference compiler or fixture", idx, reference_outcome))
        decisions = design.combine(outcome, j, d, c)
        route = outcome == "refer"
        checker_stage = "C" if j == "allow" else "M"
        decisions["matched"] = answer(checker_stage)["decision"] if route else outcome
        second = ["J", "C"] if j == "allow" else ["J"]
        paths = {
            "jev": ["J"],
            "deepseek": ["D"],
            "cascade": second,
            "verifier": ["E"],
            "residual_deepseek": ["E"] + (["D"] if route else []),
            "residual_jev": ["E"] + (["J"] if route else []),
            "matched": ["E"] + ([checker_stage] if route else []),
            "joint": ["E"] + (second if route else []),
        }
        row = {k: item[k] for k in ("id", "group", "domain", "class", "gold", "oracle")}
        row.update(
            extraction=json.dumps(p, ensure_ascii=False),
            extraction_exact=int(semantics(p) == semantics(oracle)),
            oracle_route=reference_outcome,
            oracle_reason=reference_reason,
            program_outcome=outcome,
            reason=why,
            **decisions,
        )
        for arm in ARMS:
            calls = [r for part in paths[arm] for r in by[idx, part]]
            row[arm + "_seconds"] = round(sum(r["seconds"] for r in calls), 6)
            row[arm + "_cny"] = sum(Decimal(r["estimated_cny"]) for r in calls)
            row[arm + "_calls"] = sum(r["attempts"] for r in calls)
        rows.append(row)
    return rows


def aggregate(rows, arms=ARMS):
    out = []
    for domain in ("all", *design.DOMAINS):
        group = [r for r in rows if domain == "all" or r["domain"] == domain]
        for arm in arms:
            den = Counter(r["class"] for r in group)
            pos = Counter(r["class"] for r in group if r[arm] == "allow")
            times = sorted(r[arm + "_seconds"] for r in group)
            out.append(
                {
                    "domain": domain,
                    "method": arm,
                    "n": len(group),
                    "correct": sum(r[arm] == r["gold"] for r in group),
                    "legal": den["L"],
                    "legal_allow": pos["L"],
                    "violations": den["V"],
                    "violation_allow": pos["V"],
                    "unknown": den["U"],
                    "unknown_allow": pos["U"],
                    "calls": sum(r[arm + "_calls"] for r in group),
                    "total_cny": sum(r[arm + "_cny"] for r in group),
                    "median_s": round(median(times), 3),
                    "p95_s": round(times[math.ceil(0.95 * len(times)) - 1], 3),
                }
            )
    return out


def paired(rows, arms=ARMS):
    out = []
    groups = defaultdict(list)
    for r in rows:
        groups[r["domain"], r["group"]].append(r)
    for arm in arms[:-1]:

        def changes(group, arm=arm):
            return [
                sum(
                    r["class"] == kind and r["joint"] == new and r[arm] != new
                    for r in group
                )
                for kind in ("L", "V", "U")
                for new in ("allow", "block")
            ]

        delta = changes(rows)
        rng = random.Random(329401)
        domains = [
            [changes(g) for (d, _), g in groups.items() if domain == d]
            for domain in design.DOMAINS
        ]
        draws = []
        for _ in range(5000):
            selected = [v for d in domains for v in rng.choices(d, k=len(d))]
            draws.append(sum(v[0] - v[1] for v in selected))
        draws.sort()
        out.append(
            {
                "comparator": arm,
                "legal_gained": delta[0],
                "legal_lost": delta[1],
                "new_violation": delta[2],
                "prevented_violation": delta[3],
                "new_unknown": delta[4],
                "prevented_unknown": delta[5],
                "net_legal": delta[0] - delta[1],
                "family_sensitivity_low": draws[124],
                "family_sensitivity_high": draws[4874],
            }
        )
    return out


def main(from_tables, replication):
    folder = DEST / "replication" if replication else DEST
    folder.mkdir(parents=True, exist_ok=True)
    if not from_tables:
        receipts(folder, replication)
    rows = decide(folder, replication)
    scores = aggregate(rows)
    write(folder, "decisions.csv", rows)
    write(folder, "scores.csv", scores)
    write(folder, "paired.csv", paired(rows))
    assets = design.ROOT / "paper/generated"
    (assets / ("replication.tex" if replication else "natural.tex")).write_text(
        "\n".join(
            f"{name} & {r['correct']} & {r['violation_allow']} & {r['unknown_allow']} & {r['legal_allow']} & {r['calls']} & {r['median_s']:.2f} & {r['total_cny']:.4f} "
            + r"\\"
            for name, r in zip(ENGLISH, scores[: len(ARMS)])
        )
        + "\n",
        encoding="utf8",
    )
    for r in scores[: len(ARMS)]:
        print(r)
    print("Reasons:", Counter(r["reason"] for r in rows))
    print("Extraction:", sum(r["extraction_exact"] for r in rows))
    print(
        "Joint errors:",
        [(r["id"], r["group"], r["reason"]) for r in rows if r["joint"] != r["gold"]],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-tables", action="store_true")
    opts = parser.parse_args()
    for replication in (False, True):
        main(opts.from_tables, replication)

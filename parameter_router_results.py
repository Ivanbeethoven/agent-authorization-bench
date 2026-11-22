"""Paired outcomes and marginal path costs at frozen, equal GPT quotas."""

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from statistics import median

import codex_judgments as cli
import datasets_local as ds
import parameter_router as design
from study_report import inputs, measurements

METHODS = ("deepseek", "prior_raw", "prior_calibrated", "parameters")


def write_rows(name, rows):
    with (design.DEST / (name + ".csv")).open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def collect(study):
    plans, protocol = (
        design.extra_config() if study == design.EXTRA else design.config()
    )
    billing = []
    lookup = {}
    path = defaultdict(list)
    for provider in ("deepseek", "jev"):
        frozen, raw, ledger = inputs(study, provider)
        folder = ds.ROOT / "data/study_runs" / study / provider
        if (
            frozen != plans
            or json.loads((folder / "plan.json").read_text("utf8"))["protocol"]
            != protocol
        ):
            raise ValueError("Provider frozen inputs differ")
        measured, bill = measurements(study, provider, raw, ledger)
        billing.append(bill)
        for r, c in zip(raw, measured, strict=True):
            k = r["dataset"], r["id"], r["stage"]
            path[k].append(c)
            if "answers" in r:
                if k in lookup:
                    raise ValueError("Duplicate successful response")
                lookup[k] = r
    ranks = design.allocate(plans, lookup)
    selected = design.selected_inputs(study)
    selections = {k: [r["id"] for r in p["samples"]] for k, p in selected.items()}
    folder = cli.ROOT / study
    if json.loads((folder / "selection.json").read_text("utf8")) != selections:
        raise ValueError("Different senior review manifest")
    expert = {}
    logs = ds.records(folder / "calls.jsonl")
    retries = {
        r["batch"]: r
        for r in json.loads((design.DEST / "cli_retries.json").read_text("utf8"))
        if r["study"] == study
    }
    for r in logs:
        (a,) = r["answers"]
        k = r["dataset"], a["id"]
        if (
            k in expert
            or r["model"] != cli.MODEL
            or a["decision"] not in plans[k[0]]["choices"]
        ):
            raise ValueError("Invalid independent senior result")
        retry = retries.get(r["batch"], {})
        if retry and retry["successful_start_utc"] != r["started"]:
            retry = {}
        expert[k] = (
            a["decision"],
            r["seconds"] + retry.get("elapsed_seconds", 0),
            retry.get("prior_attempts", 0),
        )
    if set(expert) != {(k, i) for k, items in selections.items() for i in items}:
        raise ValueError("Senior coverage differs from frozen selections")
    rows = []
    for name, p in plans.items():
        for r in p["samples"]:
            idx = r["id"]
            k = name, idx
            base = lookup[*k, "direct"]["answers"]["decision"]
            allow = "ACT" if name == "rules" else "allow"
            params = lookup[*k, "checks"]["answers"]
            cited = design.evidence(r["state"]).get(params["passage"], {})
            for method in METHODS:
                routed, rank, score = ranks.get((*k, method), (False, None, None))
                stages = (
                    ["direct"]
                    if method == "deepseek"
                    else ["direct", "checks", "routing"]
                    if method == "parameters"
                    else ["prior"]
                    if routed
                    else ["prior", "direct"]
                )
                calls = [c for stage in stages for c in path[*k, stage]]
                answer, t, retried = expert[k] if routed else (base, 0, 0)
                rows.append(
                    {
                        "source": name,
                        "benchmark": design.BENCH.get(name, "Jev Boundary"),
                        "id": idx,
                        "group": r["group"],
                        "method": method,
                        "reference": r["gold"],
                        "allow_label": allow,
                        "deepseek": base,
                        "decision": answer,
                        "routed": int(routed),
                        "retry": retried,
                        "rank": rank,
                        "ranking_score": json.dumps(score) if score else "",
                        "correct": int(answer == r["gold"]),
                        "false_allow": int(answer == allow != r["gold"]),
                        "legal_allow": int(answer == r["gold"] == allow),
                        "positive": int(r["gold"] == allow),
                        "fixed": int(answer == r["gold"] != base),
                        "lost": int(base == r["gold"] != answer),
                        "api_cny": sum((c["estimated_cny"] for c in calls), Decimal(0)),
                        "seconds": round(sum(c["seconds"] for c in calls) + t, 6),
                        "gpt_seconds": t,
                        "api_calls": len(calls),
                        **{
                            key: params[key] if method == "parameters" else ""
                            for key in design.DIMENSIONS
                        },
                        "cited_field": cited.get("field", "")
                        if method == "parameters"
                        else "",
                        "passage_id": params["passage"]
                        if method == "parameters"
                        else "",
                    }
                )
    times = [datetime.fromisoformat(r["started"]) for r in logs]
    clibill = {
        "calls": len(logs),
        "attempts": len(logs) + sum(entry[2] for entry in expert.values()),
        **{
            k: sum(r["usage"].get(k, 0) for r in logs)
            for k in ("input_tokens", "cached_input_tokens", "output_tokens")
        },
        "wall_seconds": round(
            (
                max(
                    t + timedelta(seconds=r["seconds"])
                    for t, r in zip(times, logs, strict=True)
                )
                - min(times)
            ).total_seconds(),
            2,
        ),
    }
    return rows, billing, clibill


def summarize(scope, rows):
    n = len(rows)
    times = sorted(r["seconds"] for r in rows)
    keys = (
        "correct",
        "false_allow",
        "legal_allow",
        "positive",
        "fixed",
        "lost",
        "api_calls",
        "api_cny",
        "routed",
        "retry",
    )
    sums = {k: sum(r[k] for r in rows) for k in keys}
    return {
        "scope": scope,
        "method": rows[0]["method"],
        "n": n,
        **sums,
        "negative": n - sums["positive"],
        "accuracy": sums["correct"] / n,
        "p50_s": round(median(times), 3),
        "p95_s": round(times[math.ceil(n * 0.95) - 1], 3),
    }


def build():
    first, bills, first_cli = collect(design.STUDY)
    extra, extra_bills, extra_cli = collect(design.EXTRA)
    rows, billing = first + extra, bills + extra_bills
    cli_bill = {
        k: first_cli[k] + extra_cli[k]
        for k in (
            "calls",
            "attempts",
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
        )
    }
    cli_bill["batch_wall_seconds"] = {
        design.STUDY: first_cli["wall_seconds"],
        design.EXTRA: extra_cli["wall_seconds"],
    }
    groups = defaultdict(list)
    for r in rows:
        labels = []
        if r["source"] != "test":
            labels.extend(["all", r["benchmark"]])
        if r["source"] != "extra":
            labels.append("initial_all")
            if r["benchmark"] == "Jev Boundary":
                labels.append("initial_jev")
        if r["source"] in ("test", "archive", "extra"):
            labels.append(r["source"])
        for label in labels:
            groups[label, r["method"]].append(r)
    scopes = [
        "all",
        *dict.fromkeys(design.BENCH.values()),
        "initial_all",
        "initial_jev",
        "test",
        "archive",
        "extra",
    ]
    scores = [summarize(s, groups[s, m]) for s in scopes for m in METHODS]
    for s in scores:
        quota = {"all": 30, "initial_all": 30, "test": 4, "extra": 4, "archive": 6}.get(
            s["scope"], 10
        )
        if s["routed"] != (0 if s["method"] == "deepseek" else quota):
            raise ValueError("Unequal review budget")
    design.DEST.mkdir(parents=True, exist_ok=True)
    write_rows("decisions", rows)
    write_rows("scores", scores)
    write_rows("provider_usage", billing)
    report = {"scores": scores, "usage": billing, "cli": cli_bill}
    (design.DEST / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, default=str, indent=2) + "\n",
        encoding="utf8",
    )
    for s in scores:
        if s["scope"] == "all":
            print(
                s["method"],
                s["correct"],
                s["false_allow"],
                s["legal_allow"],
                s["fixed"],
                s["lost"],
                s["api_cny"],
                s["p50_s"],
            )


if __name__ == "__main__":
    build()

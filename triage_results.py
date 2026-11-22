"""Evaluate reviewer selection on recorded outcomes with matched routing accounting."""

import json
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from statistics import median

import codex_judgments as cli
import datasets_local as ds
import jev_triage as design
import retained_validation as prior
from retained_results import sum_components
from shared_review import HISTORY
from study_report import inputs, measurements

DEST = ds.ROOT / "results/jev_triage"


def collect():
    plans, protocol = design.config()
    cfg, raw, ledger = inputs(design.STUDY, "jev")
    if (
        cfg != plans
        or json.loads((prior.BASE / design.STUDY / "jev/plan.json").read_text("utf8"))[
            "protocol"
        ]
        != protocol
    ):
        raise ValueError("Gate plan differs")
    costs, gatebill = measurements(design.STUDY, "jev", raw, ledger)
    gate = {}
    stagepath = defaultdict(list)
    for c, m in zip(raw, costs, strict=True):
        k = c["dataset"], c["id"]
        stagepath[k].append((m["estimated_cny"], m["seconds"]))
        if not c.get("error"):
            if k in gate:
                raise ValueError("Duplicate gate response")
            gate[k] = {**c["answers"], "prob": c["probabilities"]["route"]["senior"]}
    base = {}
    for split, study in {
        **HISTORY,
        "recent": prior.STUDY,
        "independent": design.STUDY,
    }.items():
        direct_plans, raw, bill = inputs(study, "deepseek")
        if split == "independent" and direct_plans != {
            design.FRESH: plans[design.FRESH]
        }:
            raise ValueError("Independent reviewers received different samples")
        parsed, total = measurements(study, "deepseek", raw, bill)
        if split == "independent":
            deepbill = total
        paths = defaultdict(list)
        answers = {}
        for c in parsed:
            if c["stage"] != "direct":
                continue
            name = (
                c["dataset"] if split == "independent" else split + "." + c["dataset"]
            )
            key = name, c["id"]
            paths[key].append((c["estimated_cny"], c["seconds"]))
            if not c["error"]:
                answers[key] = json.loads(c["selected"])["decision"]
        for k, a in answers.items():
            base[k] = a, *sum_components(paths[k])
    expert = {}
    new = ds.records(cli.ROOT / design.STUDY / "calls.jsonl")
    for log in [
        *ds.records(cli.ROOT / "calls.jsonl"),
        *ds.records(cli.ROOT / prior.STUDY / "calls.jsonl"),
        *new,
    ]:
        split = log["sample_split"]
        key = log["dataset"]
        if split != "triage":
            key = ("recent" if split == "retained" else split) + "." + key
        for a in log["answers"]:
            expert[key, a["id"]] = a["decision"], log["seconds"]
    output = []
    for k, p in plans.items():
        phase, name = k.split(".", 1)
        benchmark = (
            prior.NAMES[name] if name != "final_pairs" else prior.NAMES["new_pairs"]
        )
        yes = "ACT" if name == "new_delegation" else "allow"
        for row in p["samples"]:
            ix = k, row["id"]
            g = gate[ix]
            d, df, dt = base[ix]
            e, et = expert[ix]
            truth = row["gold"]
            if d not in p["choices"] or e not in p["choices"]:
                raise ValueError("Invalid decision")
            gf, gt = sum_components(stagepath[ix])
            fa_gain = int(d == yes != truth) - int(e == yes != truth)
            legal_gain = int(e == yes == truth) - int(d == yes == truth)
            output.append(
                {
                    "set": k,
                    "benchmark": benchmark,
                    "phase": {
                        "previous": "calibration",
                        "fresh": "historical",
                        "regression": "historical",
                        "recent": "recent",
                        "independent": "new",
                    }[phase],
                    "id": row["id"],
                    "group": row["group"],
                    "reference": truth,
                    "allow": yes,
                    "deepseek": d,
                    "gpt": e,
                    "deep_fee": df,
                    "deep_s": dt,
                    "gpt_s": et,
                    "gate_fee": gf,
                    "gate_s": gt,
                    **g,
                    "cell": g["factor"] + "/" + g["route"],
                    "gain": int(e == truth) - int(d == truth) + 3 * fa_gain,
                    "fa_gain": fa_gain,
                    "legal_gain": legal_gain,
                }
            )
    if len(output) != len(gate) or len(new) != 50:
        raise ValueError("Unexpected coverage")
    times = [datetime.fromisoformat(c["started"]) for c in new]
    gptbill = {
        "calls": len(new),
        "input_tokens": sum(c["usage"]["input_tokens"] for c in new),
        "output_tokens": sum(c["usage"]["output_tokens"] for c in new),
        "wall_seconds": round(
            (
                max(
                    t + timedelta(seconds=c["seconds"])
                    for t, c in zip(times, new, strict=True)
                )
                - min(times)
            ).total_seconds(),
            2,
        ),
    }
    return output, [gatebill, deepbill], gptbill


def summary(rows, method):
    def picked(r):
        if method == "gpt":
            return True
        if method == "deepseek":
            return False
        return r[method]

    flags = [picked(r) for r in rows]
    count = sum(flags)
    preds = [
        r["gpt" if flag else "deepseek"] for r, flag in zip(rows, flags, strict=True)
    ]
    fixed = lost = fa = legal = pos = recoverable = 0
    fee = Decimal(0)
    durations = []
    for r, flag, p in zip(rows, flags, preds, strict=True):
        y = r["reference"]
        d = r["deepseek"]
        yes = r["allow"]
        fixed += p == y != d
        lost += d == y != p
        recoverable += r["gpt"] == y != d
        fa += p == yes != y
        legal += p == yes == y
        pos += y == yes
        fee += (r["gate_fee"] if method in ("raw", "calibrated") else 0) + (
            0 if flag else r["deep_fee"]
        )
        durations.append(
            (r["gate_s"] if method in ("raw", "calibrated") else 0)
            + (r["gpt_s"] if flag else r["deep_s"])
        )
    # The matched random control uses the same number of escalations within each task.
    slices = defaultdict(list)
    for r, flag in zip(rows, flags, strict=True):
        slices[r["benchmark"]].append((r, flag))
    random_correct = random_fa = 0.0
    for part in slices.values():
        q = sum(flag for _, flag in part) / len(part)
        for r, _ in part:
            for model, weight in (("deepseek", 1 - q), ("gpt", q)):
                random_correct += weight * (r[model] == r["reference"])
                random_fa += weight * (r[model] == r["allow"] != r["reference"])
    return {
        "method": method,
        "n": len(rows),
        "gpt_calls": count,
        "correct": sum(p == r["reference"] for r, p in zip(rows, preds, strict=True)),
        "false_allows": fa,
        "positive": pos,
        "negative": len(rows) - pos,
        "released": legal,
        "fixed": fixed,
        "lost": lost,
        "recoverable": recoverable,
        "random_correct": round(random_correct, 2),
        "random_fa": round(random_fa, 2),
        "api_cny": str(fee),
        "p50_s": round(median(durations), 3),
    }


def build():
    rows, usage, gptbill = collect()
    rules = {}
    for name in sorted({r["benchmark"] for r in rows}):
        train = [
            r for r in rows if r["phase"] == "calibration" and r["benchmark"] == name
        ]
        values, threshold = design.fit(train)
        rules[name] = {"values": values, "threshold": threshold, "cases": len(train)}
        for r in (r for r in rows if r["benchmark"] == name):
            r["raw"] = r["route"] == "senior"
            r["calibrated"] = (
                threshold is not None
                and (values.get(r["cell"], 0), r["prob"]) >= threshold
            )
    scores = []
    for phase in ("calibration", "historical", "recent", "new"):
        phase_rows = [r for r in rows if r["phase"] == phase]
        for key in ["all", *sorted({r["benchmark"] for r in phase_rows})]:
            group = [r for r in phase_rows if key == "all" or r["benchmark"] == key]
            for method in ("deepseek", "raw", "calibrated", "gpt"):
                scores.append(
                    {"phase": phase, "benchmark": key, **summary(group, method)}
                )
    DEST.mkdir(parents=True, exist_ok=True)
    import csv

    for name, table in (
        ("decisions", rows),
        ("scores", scores),
        ("provider_usage", usage),
    ):
        with (DEST / (name + ".csv")).open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            w = csv.DictWriter(stream, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    report = {"scores": scores, "rules": rules, "usage": usage, "cli": gptbill}
    (DEST / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, default=str, indent=2) + "\n",
        encoding="utf8",
    )
    print(
        [
            (
                r["method"],
                r["gpt_calls"],
                r["correct"],
                r["false_allows"],
                r["released"],
            )
            for r in scores
            if r["phase"] == "new" and r["benchmark"] == "all"
        ]
    )


if __name__ == "__main__":
    build()

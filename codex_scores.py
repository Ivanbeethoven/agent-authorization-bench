"""Join blinded Codex outputs to the local references and model decisions."""

import math
from decimal import Decimal
from statistics import median

import codex_judgments as cli
import datasets_local as ds
import parameter_external as ext
import parameter_scores as ps
import reply_results as study


def evaluate():
    inputs = {b[0]: b for b in cli.batches()}
    calls = ds.records(cli.ROOT / "calls.jsonl")
    if len(calls) != len(inputs) or {r["batch"] for r in calls} != inputs.keys():
        raise ValueError("Missing or duplicate Codex batches")
    previous = {
        (
            "regression" if r["set"] == "final_pairs" else "previous",
            r["set"],
            r["id"],
        ): r
        for r in study.original_errors()
    }
    for r in ps.read_csv(study.DEST / "decisions.csv"):
        if r["set"] != "final_pairs":
            previous["fresh", r["set"], r["id"]] = r
    decisions, billing = [], []
    costs = {
        (r["set"], r["id"], r["method"]): r
        for r in ps.read_csv(study.DEST / "method_costs.csv")
    }
    new_by_id = {
        (r["set"], r["id"]): r for r in ps.read_csv(study.DEST / "decisions.csv")
    }
    for c in sorted(calls, key=lambda c: c["batch"]):
        key, alias, name, _, choices, rows = inputs[c["batch"]]
        if c["model"] != cli.MODEL or [r["id"] for r in rows] != [
            a["id"] for a in c["answers"]
        ]:
            raise ValueError("Unexpected Codex model or sample")
        bill = {
            "split": alias,
            "set": name,
            "batch": key,
            "cases": len(rows),
            "seconds": c["seconds"],
            "started": c["started"],
            **c["usage"],
        }
        billing.append(bill)
        for r, a in zip(rows, c["answers"], strict=True):
            old = previous[alias, name, r["id"]]
            if old["reference"] != r["gold"] or a["decision"] not in choices:
                raise ValueError("Mismatched reference or label")
            decisions.append(
                {
                    "split": alias,
                    "set": name,
                    "id": r["id"],
                    "group": r["group"],
                    "reference": r["gold"],
                    "allow_label": ext.translate(name, "allow"),
                    "codex_sol": a["decision"],
                    "deepseek_direct": old["deepseek_direct"],
                    "joint_adaptive": old["joint_adaptive"],
                    "batch": key,
                    "sol_arbitration": "",
                    "consulted": "",
                }
            )
            if alias != "previous":
                change = new_by_id[name, r["id"]]["inverse"] != old["deepseek_direct"]
                decisions[-1].update(
                    consulted=int(change),
                    sol_arbitration=a["decision"] if change else old["deepseek_direct"],
                )
    scores = []
    for alias, name in sorted({(r["split"], r["set"]) for r in decisions}):
        group = [r for r in decisions if (r["split"], r["set"]) == (alias, name)]
        b = [r for r in billing if (r["split"], r["set"]) == (alias, name)]
        times = sorted(r["seconds"] for r in b)
        s = ps.summary(group, "codex_sol")
        scores.append(
            {
                "split": alias,
                "set": name,
                "method": "codex_sol",
                **s,
                "accuracy": round(s["correct"] / s["cases"], 6),
                "corrected_vs_direct": sum(
                    r["codex_sol"] == r["reference"] != r["deepseek_direct"]
                    for r in group
                ),
                "lost_vs_direct": sum(
                    r["deepseek_direct"] == r["reference"] != r["codex_sol"]
                    for r in group
                ),
                "calls": len(b),
                "p50_s": round(median(times), 3),
                "p95_s": round(times[math.ceil(0.95 * len(times)) - 1], 3),
                "max_batch_size": max(r["cases"] for r in b),
                "cost_cny": "",
                **{
                    k: sum(r[k] for r in b)
                    for k in ("input_tokens", "cached_input_tokens", "output_tokens")
                },
            }
        )
        if alias != "previous":
            routed = [r for r in group if r["consulted"]]
            timing = {r["batch"]: r["seconds"] for r in b}
            n = len(group)
            amount = Decimal(0)
            duration = []
            for r in group:
                both = [
                    costs[name, r["id"], arm] for arm in ("inverse", "deepseek_direct")
                ]
                amount += sum(Decimal(m["cost_cny"]) for m in both)
                duration.append(
                    sum(float(m["seconds"]) for m in both)
                    + (timing[r["batch"]] if r["consulted"] else 0)
                )
            duration.sort()
            s = ps.summary(group, "sol_arbitration")
            scores.append(
                {
                    "split": alias,
                    "set": name,
                    "method": "sol_arbitration",
                    **s,
                    "accuracy": round(s["correct"] / n, 6),
                    "corrected_vs_direct": sum(
                        r["sol_arbitration"] == r["reference"] != r["deepseek_direct"]
                        for r in group
                    ),
                    "lost_vs_direct": sum(
                        r["deepseek_direct"] == r["reference"] != r["sol_arbitration"]
                        for r in group
                    ),
                    "calls": 2 * n + len(routed),
                    "p50_s": round(median(duration), 3) if alias == "fresh" else "",
                    "p95_s": round(duration[math.ceil(0.95 * n) - 1], 3)
                    if alias == "fresh"
                    else "",
                    "max_batch_size": "",
                    "cost_cny": "",
                    "input_tokens": "",
                    "cached_input_tokens": "",
                    "output_tokens": "",
                    "api_cost_cny": str(amount),
                    "consulted": len(routed),
                }
            )
    for r in scores:
        r.setdefault("api_cost_cny", "")
        r.setdefault("consulted", "")
    return decisions, billing, scores


if __name__ == "__main__":
    for filename, rows in zip(
        ("codex_decisions", "codex_calls", "codex_scores"), evaluate(), strict=True
    ):
        study.write_csv(filename, rows)

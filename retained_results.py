"""Score blinded decisions and account for shared calls on the frozen new cohorts."""

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from random import Random
from statistics import median

import codex_judgments as cli
import datasets_local as ds
import parameter_external as native
import parameter_scores as ps
import reply_results
import reply_review
import retained_validation as design
import shared_review
from study_report import inputs, measurements

DEST = design.DEST
MAIN = tuple(design.LOCKED_ROUTE)
SOURCE = {
    "new_pairs": "自建授权配对",
    "new_delegation": "自建委托配对",
    "new_boundary": "公开 test",
    "disputed": "公开争议题 · 二分类",
}


def save(name, rows):
    with (DEST / (name + ".csv")).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def get_data(plans, protocol):
    raw_answers = defaultdict(dict)
    all_calls = defaultdict(list)
    usage = []
    starts, ends = [], []
    for provider in ("deepseek", "jev", "aliyun"):
        seen, raw, ledger = inputs(design.STUDY, provider)
        expected = {
            k: p
            for k, p in plans.items()
            if provider != "aliyun" or "aliyun_direct" in protocol["eligible"][k]
        }
        snap = json.loads(
            (design.BASE / design.STUDY / provider / "plan.json").read_text("utf8")
        )
        if seen != expected or snap["protocol"] != protocol:
            raise ValueError("Frozen evaluation inputs differ")
        calls, total = measurements(design.STUDY, provider, raw, ledger)
        usage.append(total)
        starts.append(datetime.fromisoformat(ledger["start"]))
        ends.append(datetime.fromisoformat(ledger["end"]))
        for c in calls:
            ix = provider, c["dataset"], c["id"]
            all_calls[ix].append(c)
            if not c["error"]:
                if c["stage"] in raw_answers[ix]:
                    raise ValueError("Repeated successful call")
                raw_answers[ix][c["stage"]] = json.loads(c["selected"])
    gpt = {}
    logs = ds.records(cli.ROOT / design.STUDY / "calls.jsonl")
    for c in logs:
        name = c["dataset"]
        if len(c["answers"]) != 1 or c["model"] != cli.MODEL:
            raise ValueError("Unexpected Codex batch")
        a = c["answers"][0]
        key = (name, a["id"])
        if key in gpt or a["decision"] not in plans[name]["choices"]:
            raise ValueError("Invalid GPT output")
        gpt[key] = (a["decision"], c["seconds"])
        start = datetime.fromisoformat(c["started"])
        starts.append(start)
        ends.append(start + timedelta(seconds=c["seconds"]))
    if set(gpt) != {(k, r["id"]) for k, p in plans.items() for r in p["samples"]}:
        raise ValueError("Incomplete single-request GPT results")
    gpt_times = [datetime.fromisoformat(r["started"]) for r in logs]
    cli_usage = {
        "calls": len(logs),
        "input_tokens": sum(r["usage"]["input_tokens"] for r in logs),
        "output_tokens": sum(r["usage"]["output_tokens"] for r in logs),
        "cached_input_tokens": sum(r["usage"]["cached_input_tokens"] for r in logs),
        "wall_seconds": round(
            (
                max(
                    t + timedelta(seconds=r["seconds"])
                    for t, r in zip(gpt_times, logs, strict=True)
                )
                - min(gpt_times)
            ).total_seconds(),
            2,
        ),
    }
    return (
        raw_answers,
        all_calls,
        gpt,
        usage,
        cli_usage,
        round((max(ends) - min(starts)).total_seconds(), 2),
    )


def predict(plans, protocol, answers, calls, gpt):
    details = []
    for name, plan in plans.items():
        which = "new_delegation" if name == "disputed" else name
        yes = native.translate(which, "allow")
        for r in plan["samples"]:
            idx = r["id"]
            j, d = (answers[p, name, idx] for p in ("jev", "deepseek"))
            direct = d["direct"]["decision"]
            test = d["reply_test"]
            inverse = native.translate(which, reply_review.ACTION[test["reply"]])
            sol, cli_seconds = gpt[name, idx]

            def path(provider, stage, name=name, idx=idx):
                part = [v for v in calls[provider, name, idx] if v["stage"] == stage]
                if sum(not c["error"] for c in part) != 1:
                    raise ValueError("Missing successful stage")
                return (
                    sum(c["estimated_cny"] for c in part),
                    sum(c["seconds"] for c in part),
                    len(part),
                    0,
                )

            dp, jp, review = (
                path("deepseek", "direct"),
                path("jev", "direct"),
                path("deepseek", "reply_test"),
            )
            codex = (Decimal(0), cli_seconds, 1, 1)
            records = {
                "deepseek_direct": (direct, [dp]),
                "jev_direct": (j["direct"]["decision"], [jp]),
                "codex_sol": (sol, [codex]),
                "inverse": (inverse, [review]),
                "sol_arbitration": (
                    sol if inverse != direct else direct,
                    [dp, review] + ([codex] if inverse != direct else []),
                ),
            }
            if name in ("new_pairs", "new_boundary"):
                sym = reply_review.decisions(
                    which, direct, direct, j["reply_test"], test
                )["symmetric"]
                consult = (
                    inverse != direct
                    and reply_review.CONSISTENT[test["reply"]] == test["obstacle"]
                )
                records["symmetric"] = (
                    sym,
                    [dp, review] + ([path("jev", "reply_test")] if consult else []),
                )
            if name == "new_pairs":
                old = reply_results.legacy(which, r, j, d)
                outcome = {
                    **old,
                    "id": idx,
                    "set": name,
                    "group": r["group"],
                    "allow_label": yes,
                    "reference": r["gold"],
                }
                summary = ps.summarize_split(
                    [outcome], "joint_adaptive", calls, {"joint": "version"}
                )
                records["joint_adaptive"] = (
                    old["joint_adaptive"],
                    [(summary["cost_cny"], summary["p50_s"], summary["calls"], 0)],
                )
            if which == "new_delegation":
                records["precedent"] = (
                    d["precedent"]["decision"],
                    [path("deepseek", "precedent")],
                )
                records["commitment"] = (
                    shared_review.commitment_review(direct, d["effect"]),
                    [dp]
                    + (
                        [path("deepseek", "grant"), path("deepseek", "effect")]
                        if direct == yes
                        else []
                    ),
                )
            if "aliyun_direct" in protocol["eligible"][name]:
                records["aliyun_direct"] = (
                    answers["aliyun", name, idx]["direct"]["decision"],
                    [path("aliyun", "direct")],
                )
            for arm in protocol["eligible"][name]:
                label, route = records[arm]
                if label not in plan["choices"]:
                    raise ValueError("Wrong label space")
                gold = r["gold"]
                pred = label if name != "disputed" or label == yes else "NON_ACT"
                fee, seconds, n, g = sum_components(route)
                details.append(
                    {
                        "set": name,
                        "id": idx,
                        "group": r["group"],
                        "method": arm,
                        "reference": gold,
                        "native_prediction": label,
                        "prediction": pred,
                        "allow_label": yes,
                        "correct": int(pred == gold),
                        "false_allow": int(pred == yes != gold),
                        "released": int(pred == gold == yes),
                        "api_cny": fee,
                        "seconds": round(seconds, 6),
                        "calls": n,
                        "gpt_calls": g,
                    }
                )
    return details


def sum_components(parts):
    return tuple(sum(col) for col in zip(*parts, strict=True))


def accuracy_interval(rows, baseline):
    blocks = defaultdict(list)
    for r in rows:
        blocks[r["group"]].append(r["correct"] - baseline[r["id"]]["correct"])
    stats = [(sum(b), len(b)) for b in blocks.values()]
    rand = Random(48011)
    deltas = []
    for _ in range(2000):
        total, n = sum_components(rand.choices(stats, k=len(stats)))
        deltas.append(100 * total / n)
    deltas.sort()
    return round(deltas[49], 2), round(deltas[1949], 2)


def earlier_scores(name, arm):
    prior = {
        "new_pairs": ("regression.final_pairs", "previous.new_pairs", "pilot.authgate"),
        "new_delegation": (
            "fresh.new_delegation",
            "previous.new_delegation",
            "pilot.delegation",
        ),
        "new_boundary": ("fresh.new_boundary", "previous.new_boundary", "pilot.jev"),
    }
    if name not in prior:
        return {}
    base = ds.ROOT / "results/shared_review"
    index = {(r["batch"], r["method"]): r for r in design.read(base / "scores.csv")}
    kept = {
        (r["batch"], r["method"])
        for r in design.read(base / "strategy_selection.csv")
        if r["status"] == "keep"
    }
    b = next(b for b in prior[name] if (b, arm) in kept)
    a, d = index[b, arm], index[b, "deepseek_direct"]
    return {
        "prior_batch": b,
        "old_acc_pp": round(
            100 * (int(a["correct"]) - int(d["correct"])) / int(a["cases"]), 2
        ),
        "old_fa_pp": round(
            100
            * (int(a["false_allows"]) - int(d["false_allows"]))
            / int(a["negative"]),
            2,
        ),
        "old_p50_delta": round(float(a["p50_s"]) - float(d["p50_s"]), 3)
        if a["p50_s"] and d["p50_s"]
        else "",
        "old_fee_delta": str(
            100 * (Decimal(a["cost_cny"]) - Decimal(d["cost_cny"])) / int(a["cases"])
        )
        if a["cost_cny"] and d["cost_cny"]
        else "",
    }


def summarize(details, protocol):
    scores = []
    for name, methods in protocol["eligible"].items():
        cohort = [r for r in details if r["set"] == name]
        base = {r["id"]: r for r in cohort if r["method"] == "deepseek_direct"}
        for m in methods:
            rows = [r for r in cohort if r["method"] == m]
            if len(rows) != len(base) or {r["id"] for r in rows} != set(base):
                raise ValueError("Different paired samples")
            n = len(rows)

            def total(field, records=rows):
                return sum(r[field] for r in records)

            positive = sum(r["reference"] == r["allow_label"] for r in rows)
            sorted_times = sorted(r["seconds"] for r in rows)
            better = sum(r["correct"] > base[r["id"]]["correct"] for r in rows)
            worse = sum(r["correct"] < base[r["id"]]["correct"] for r in rows)
            fa_change = total("false_allow") - sum(
                r["false_allow"] for r in base.values()
            )
            lo, hi = accuracy_interval(rows, base)
            scores.append(
                {
                    "set": name,
                    "batch": "validation." + name,
                    "benchmark": design.NAMES[name],
                    "source": SOURCE[name],
                    "method": m,
                    "cases": n,
                    "groups": len({r["group"] for r in rows}),
                    "correct": total("correct"),
                    "accuracy": total("correct") / n,
                    "negative": n - positive,
                    "positive": positive,
                    "false_allows": total("false_allow"),
                    "released": total("released"),
                    "new_acc_pp": round(100 * (better - worse) / n, 2),
                    "new_fa_pp": round(100 * fa_change / (n - positive), 2),
                    "fixed": better,
                    "lost": worse,
                    "ci_low": lo,
                    "ci_high": hi,
                    "cost_cny": str(total("api_cny")) if not total("gpt_calls") else "",
                    "api_cny": str(total("api_cny")),
                    "gpt_calls": total("gpt_calls"),
                    "calls": total("calls"),
                    "p50_s": round(median(sorted_times), 3),
                    "p95_s": round(sorted_times[math.ceil(0.95 * n) - 1], 3),
                    "prior_batch": "",
                    "old_acc_pp": "",
                    "old_fa_pp": "",
                    "old_p50_delta": "",
                    "old_fee_delta": "",
                    **earlier_scores(name, m),
                }
            )
    return scores


def rank(row):
    return (
        -row["accuracy"],
        row["fa_rate"],
        -row["release_rate"],
        int(row["cost_cny"] == ""),
        Decimal(row["cost_cny"]) if row["cost_cny"] != "" else Decimal("Infinity"),
        row["p50_s"],
        row["method"],
    )


def winners(scores, details, eligible):
    idx = {(s["set"], s["method"]): s for s in scores}

    def passes(r, d):
        return (
            r["correct"] >= d["correct"]
            and r["false_allows"] <= d["false_allows"]
            and r["released"] >= d["released"]
        )

    best = {}
    for k in MAIN:
        rows = [
            {
                **r,
                "fa_rate": r["false_allows"] / r["negative"],
                "release_rate": r["released"] / r["positive"],
            }
            for (n, _), r in idx.items()
            if n == k and passes(r, idx[k, "deepseek_direct"])
        ]
        best[k] = min(rows, key=rank)["method"]
    common = set.intersection(*(set(eligible[k]) for k in MAIN))
    options = {m: dict.fromkeys(MAIN, m) for m in sorted(common)}
    options["fixed_router"] = design.LOCKED_ROUTE

    def summarize_route(m, route):
        rows = [idx[k, route[k]] for k in MAIN]
        times = [
            r["seconds"]
            for r in details
            if r["set"] in route and r["method"] == route[r["set"]]
        ]
        g = sum(r["gpt_calls"] for r in rows)
        api = sum(Decimal(r["api_cny"]) for r in rows)
        return {
            "method": m,
            "accuracy": sum(r["accuracy"] for r in rows) / 3,
            "correct": sum(r["correct"] for r in rows),
            "fa_rate": sum(r["false_allows"] / r["negative"] for r in rows) / 3,
            "false_allows": sum(r["false_allows"] for r in rows),
            "release_rate": sum(r["released"] / r["positive"] for r in rows) / 3,
            "released": sum(r["released"] for r in rows),
            "passed": all(passes(r, idx[r["set"], "deepseek_direct"]) for r in rows),
            "cost_cny": "" if g else str(api),
            "api_cny": str(api),
            "gpt_calls": g,
            "p50_s": round(median(times), 3),
        }

    combined = [summarize_route(m, route) for m, route in options.items()]
    winner = min((r for r in combined if r["passed"]), key=rank)["method"]
    return best, combined, winner, summarize_route("posthoc_per_task", best)


def build():
    plans, protocol = design.config()
    a, c, g, usage, cli_usage, wall = get_data(plans, protocol)
    details = predict(plans, protocol, a, c, g)
    scores = summarize(details, protocol)
    best, combined, winner, posthoc = winners(scores, details, protocol["eligible"])
    DEST.mkdir(exist_ok=True, parents=True)
    save("decisions", details)
    save("scores", scores)
    save("provider_usage", usage)
    save("plans_comparison", combined)
    data = {
        "scores": scores,
        "best": best,
        "combined": combined,
        "winner": winner,
        "posthoc": posthoc,
        "route": design.LOCKED_ROUTE,
        "usage": usage,
        "cli": cli_usage,
        "wall": wall,
        "groups": {
            k: len({r["group"] for r in p["samples"]}) for k, p in plans.items()
        },
        "pool": protocol["population"],
    }
    (DEST / "report.json").write_text(
        json.dumps(data, default=str, ensure_ascii=False, indent=2) + "\n",
        encoding="utf8",
    )
    print(
        "Winner:",
        winner,
        "per task:",
        best,
        "known API CNY:",
        sum(r["estimated_cny"] for r in usage),
    )


if __name__ == "__main__":
    build()

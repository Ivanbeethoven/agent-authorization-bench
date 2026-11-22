"""Paired, family-resampled uncertainty and equal-volume controls from published decisions."""

from collections import defaultdict
from random import Random

import numpy as np

import full_permission as design
import numeric_permissions as numeric
from full_permission_results import observations, read, write

SEED = 20261123
REPS = 10000


def allowed(r, method):
    return r[method] == r["allow_label"]


def counts(r, first, second):
    legal = r["legal"] == "True"
    prohibited = r["reference"] in ("deny", "REFUSE")
    delta = int(allowed(r, first)) - int(allowed(r, second))
    return np.array(
        [
            int(prohibited),
            -delta * prohibited,
            int(not legal),
            -delta * (not legal),
            int(legal),
            delta * legal,
            int(bool(r["reference"])),
            (int(r[first] == r["reference"]) - int(r[second] == r["reference"]))
            * bool(r["reference"]),
        ],
        dtype=int,
    )


def bootstrap(rows, first, second):
    rng = np.random.default_rng(SEED)
    pooled = np.zeros((REPS, 8), dtype=np.int64)
    total = np.zeros(8, dtype=np.int64)
    for scope in design.PRIMARY:
        by_family = defaultdict(lambda: np.zeros(8, dtype=int))
        for r in rows:
            if r["scope"] == scope:
                by_family[r["family"]] += counts(r, first, second)
        array = np.stack(list(by_family.values()))
        # Keep repeated clusters as repeated occurrences and preserve pairing across methods.
        weights = rng.multinomial(
            len(array), np.full(len(array), 1 / len(array)), size=REPS
        )
        pooled += weights @ array
        total += array.sum(axis=0)
    intervals = []
    for i, name in enumerate(
        ("prohibited_stop", "nonexecution_recall", "legal_release", "native_accuracy")
    ):
        denominator, diff = pooled[:, 2 * i], pooled[:, 2 * i + 1]
        valid = denominator > 0
        lo, hi = np.percentile(100 * diff[valid] / denominator[valid], [2.5, 97.5])
        intervals.append(
            {
                "first": first,
                "second": second,
                "metric": name,
                "delta_pp": 100 * total[2 * i + 1] / total[2 * i],
                "lo_pp": float(lo),
                "hi_pp": float(hi),
                "valid_draws": int(valid.sum()),
                "seed": SEED,
            }
        )
    return intervals


def diagnostics():
    rows = read("decisions.csv")
    output, controls, pairs, ties = [], [], [], []
    for competitor in ("jev", "deepseek"):
        output.extend(bootstrap(rows, "joint", competitor))
    for scope in design.COUNTS:
        group = [r for r in rows if r["scope"] == scope]
        initial = [r for r in group if allowed(r, "jev")]
        keep = sum(allowed(r, "joint") for r in group)
        # Randomized tie-break independent of IDs' lexical content and of reference labels.
        shuffled = sorted(initial, key=lambda r: r["id"])
        Random(SEED).shuffle(shuffled)
        ranked = sorted(
            shuffled, key=lambda r: float(r["allow_probability"]), reverse=True
        )
        if keep:
            threshold = float(ranked[keep - 1]["allow_probability"])
            above = [r for r in initial if float(r["allow_probability"]) > threshold]
            tied = [r for r in initial if float(r["allow_probability"]) == threshold]
            extra = keep - len(above)
            fixed = sum(r["legal"] == "False" for r in above)
            wrong = sum(r["legal"] == "False" for r in tied)
            ties.append(
                {
                    "scope": scope,
                    "cutoff": threshold,
                    "tied": len(tied),
                    "selected_from_tie": extra,
                    "min_false": fixed + max(0, extra - (len(tied) - wrong)),
                    "max_false": fixed + min(extra, wrong),
                    "expected_false": fixed + extra * wrong / len(tied),
                }
            )
        for name, selected in (
            ("cascade", [r for r in initial if allowed(r, "joint")]),
            ("confidence", ranked[:keep]),
            ("two_direct", [r for r in initial if allowed(r, "deepseek")]),
        ):
            controls.append(
                {
                    "scope": scope,
                    "method": name,
                    "released": len(selected),
                    "false_allow": sum(r["legal"] == "False" for r in selected),
                    "legal_allow": sum(r["legal"] == "True" for r in selected),
                    "prohibited_allow": sum(
                        r["reference"] in ("deny", "REFUSE") for r in selected
                    ),
                }
            )
        controls.append(
            {
                "scope": scope,
                "method": "random_expected",
                "released": keep,
                "false_allow": keep
                * sum(r["legal"] == "False" for r in initial)
                / len(initial),
                "legal_allow": keep
                * sum(r["legal"] == "True" for r in initial)
                / len(initial),
                "prohibited_allow": keep
                * sum(r["reference"] in ("deny", "REFUSE") for r in initial)
                / len(initial),
            }
        )
        if scope == "authgate":
            for cat in ("all", *sorted({r["subset"] for r in group})):
                families = defaultdict(list)
                for r in group:
                    if cat == "all" or cat == r["subset"]:
                        families[r["family"]].append(r)
                if any(
                    len(g) != 2 or {r["reference"] for r in g} != {"allow", "block"}
                    for g in families.values()
                ):
                    raise ValueError("Incomplete contrast pairs")
                for m in ("jev", "joint", "deepseek"):
                    pairs.append(
                        {
                            "category": cat,
                            "method": m,
                            "pairs": len(families),
                            "both_correct": sum(
                                all(r[m] == r["reference"] for r in g)
                                for g in families.values()
                            ),
                            "both_allowed": sum(
                                all(allowed(r, m) for r in g) for g in families.values()
                            ),
                            "both_blocked": sum(
                                all(not allowed(r, m) for r in g)
                                for g in families.values()
                            ),
                        }
                    )
    write("paired_intervals.csv", output)
    write("controls.csv", controls)
    write("contrast_pairs.csv", pairs)
    write("confidence_ties.csv", ties)


def numeric_results():
    jev, jb = observations(numeric.DIRECT, "jev")
    deep, db = observations(numeric.DIRECT, "deepseek")
    reviews, rb = observations(numeric.REVIEW, "deepseek")
    rows = []
    for r in numeric.config()["numeric"]["samples"]:
        k = "numeric", r["id"]
        j = jev[k][0]["answers"]["decision"]
        d = deep[k][0]["answers"]["decision"]
        if (k in reviews) != (j == "allow"):
            raise ValueError("Different numeric routing")
        final = reviews[k][0]["answers"]["decision"] if k in reviews else j
        verified = numeric.verify_condition(r["state"])
        if verified is None or verified != (r["gold"] == "allow"):
            raise ValueError("Independent numeric reference mismatch")
        rows.append(
            {
                "id": r["id"],
                "kind": r["kind"],
                "group": r["group"],
                "margin": r["margin"],
                "reference": r["gold"],
                "jev": j,
                "deepseek": d,
                "joint": final,
                "checked_release": "allow"
                if final == "allow" and verified
                else "block",
            }
        )
    write("numeric_decisions.csv", rows)
    write("numeric_usage.csv", [jb, db, rb])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-tables",
        action="store_true",
        help="Recompute controls and intervals from published CSVs without raw API logs.",
    )
    args = parser.parse_args()
    diagnostics()
    if not args.from_tables:
        numeric_results()

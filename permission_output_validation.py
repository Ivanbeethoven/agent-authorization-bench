"""Check original output-weight replay against direct constrained one-token decoding."""

import argparse
import csv
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from permission_features import DEFAULT_TEMPLATE, OUT, bounded, frozen, verify_server
from permission_head_results import ORIGINAL, read_csv, write
from permission_training_data import records


def direct(base, template, state):
    start = time.perf_counter()
    prompts, coverage = bounded(template, state, base)
    if not prompts:
        return 0, coverage, time.perf_counter() - start
    result = frozen.request(
        base,
        "/completion",
        {
            "prompt": prompts[0],
            "n_predict": 1,
            "temperature": 0,
            "seed": 0,
            "grammar": 'root ::= "A" | "B"',
            "cache_prompt": True,
            "stream": False,
        },
    )
    answer = result["content"].strip()
    if result.get("truncated") or answer not in ("A", "B"):
        raise ValueError("Invalid one-token control")
    return int(answer == "A"), coverage, time.perf_counter() - start


def run(port, workers, all_requests, template_path=DEFAULT_TEMPLATE):
    base = f"http://127.0.0.1:{port}"
    verify_server(base)
    tpl = frozen.render_template(template_path)
    groups = defaultdict(list)
    for split in ("challenge", "xaab", "toolsafe"):
        for key, group, state, _ in records(split):
            groups[split if split != "toolsafe" else group].append((split, key, state))
    cases = []
    rng = random.Random(112425)
    for group, members in sorted(groups.items()):
        picked = members if all_requests else rng.sample(members, 48)
        cases.extend((group, split, key, state) for split, key, state in picked)
    suffix = "all" if all_requests else "sample"
    path = OUT / f"native_original_{suffix}.csv"
    previous = read_csv(path) if path.exists() else []
    saved = {(r["split"], r["id"]) for r in previous}
    pending = [r for r in cases if (r[1], r[2]) not in saved]

    def call(case):
        group, split, key, state = case
        a, c, t = direct(base, tpl, state)
        return {
            "group": group,
            "split": split,
            "id": key,
            "allow": a,
            "coverage": c,
            "seconds": round(t, 6),
        }

    with (
        ThreadPoolExecutor(max_workers=workers) as pool,
        path.open("a", newline="", encoding="utf-8-sig") as f,
    ):
        writer = csv.DictWriter(
            f, fieldnames=("group", "split", "id", "allow", "coverage", "seconds")
        )
        if not previous:
            writer.writeheader()
        for i in range(0, len(pending), 32):
            writer.writerows(pool.map(call, pending[i : i + 32]))
            f.flush()
    rows = read_csv(path)
    if {(r["split"], r["id"]) for r in rows} != {
        (split, key) for _, split, key, _ in cases
    } or len(rows) != len(cases):
        raise ValueError("Incomplete one-token controls")
    predicted = {
        split: {r["id"]: r for r in read_csv(OUT / f"{split}_predictions.csv")}
        for split in ("challenge", "xaab", "toolsafe")
    }
    agreement = defaultdict(list)
    for r in rows:
        replay = predicted[r["split"]][r["id"]]
        if r["coverage"] != replay["coverage"]:
            raise ValueError("Different context coverage")
        agreement[r["group"]].append(r["allow"] == replay[ORIGINAL])
    audit = [{"group": g, "agree": sum(v), "n": len(v)} for g, v in agreement.items()]
    write(OUT / f"external_agreement_{suffix}.csv", audit)
    print(audit)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--port", type=int, default=18384)
    cli.add_argument("--workers", type=int, default=2)
    cli.add_argument("--all", action="store_true")
    cli.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    args = cli.parse_args()
    run(args.port, args.workers, args.all, args.template)

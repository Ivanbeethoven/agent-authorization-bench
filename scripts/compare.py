"""Compare completed Jev and Model Studio decisions on identical input manifests."""

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def predictions(folder):
    with (folder / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = {}
        for row in csv.DictReader(stream):
            key = (row["benchmark"], row["sample"])
            if key in rows or row["error"] or not row["prediction"]:
                raise ValueError(f"Duplicate or incomplete prediction: {key}")
            rows[key] = row
        return rows


def manifest(provider_folder, name):
    path = ROOT / "data" / provider_folder / f"{name}.json"
    return json.loads(path.read_text(encoding="utf8"))


def compare():
    jev = predictions(RESULTS)
    aliyun = predictions(RESULTS / "aliyun")
    if not jev or jev.keys() != aliyun.keys():
        raise ValueError("Providers did not score the same sample IDs.")

    counts = {}
    paired = []
    for dataset in dict.fromkeys(name for name, _ in jev):
        first = manifest("batches", dataset)
        second = manifest("aliyun-batches", dataset)
        for field in ("samples", "instructions", "choices"):
            if first[field] != second[field]:
                raise ValueError(f"Different {field} for {dataset}")
        cases = first["samples"]
        if {case["id"] for case in cases} != {i for name, i in jev if name == dataset}:
            raise ValueError(
                f"Sample manifest does not match predictions for {dataset}"
            )
        count = Counter()
        for case in cases:
            key = (dataset, case["id"])
            left, right = jev[key], aliyun[key]
            label = case["gold"]
            if left["reference"] != label or right["reference"] != label:
                raise ValueError(f"Reference label disagreement: {key}")
            if (
                left["prediction"] not in first["choices"]
                or right["prediction"] not in first["choices"]
            ):
                raise ValueError(f"Invalid prediction: {key}")
            jev_ok = left["prediction"] == label
            aliyun_ok = right["prediction"] == label
            same = left["prediction"] == right["prediction"]
            count["samples"] += 1
            count["jev_correct"] += jev_ok
            count["aliyun_correct"] += aliyun_ok
            count["same_decision"] += same
            count["only_jev_correct"] += jev_ok and not aliyun_ok
            count["only_aliyun_correct"] += aliyun_ok and not jev_ok
            paired.append([*key, label, left["prediction"], right["prediction"]])
        counts[dataset] = count

    with (RESULTS / "paired_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["benchmark", "sample", "reference", "jev", "aliyun"])
        writer.writerows(paired)
    fields = [
        "samples",
        "jev_correct",
        "aliyun_correct",
        "same_decision",
        "only_jev_correct",
        "only_aliyun_correct",
    ]
    with (RESULTS / "comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["benchmark", *fields])
        for dataset, count in counts.items():
            writer.writerow([dataset, *(count[field] for field in fields)])


if __name__ == "__main__":
    compare()

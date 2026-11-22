"""Build the self-contained, filterable authorization experiment report."""

import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from statistics import median

import datasets_local as ds
import parameter_review as pr
from parameter_scores import METHODS, read_csv, save

GROUPS = (
    "final_pairs",
    "new_pairs",
    "new_boundary",
    "new_delegation",
    "version_pairs",
    "old_pairs",
)


def changed(rows):
    out = []
    for name in GROUPS:
        data = [r for r in rows if r["set"] == name]
        for arm in (
            "jev_new",
            "joint_new",
            "jev_all",
            "joint_all",
            "jev_adaptive",
            "joint_adaptive",
        ):
            base = arm.split("_", 1)[0] + "_previous"
            eligible = [
                r
                for r in data
                if r["reference"] == r["allow_label"] and r[base] == r["allow_label"]
            ]
            recovered = [r for r in eligible if r["risk_tree"] != r["allow_label"]]
            bad = [
                r
                for r in data
                if r["reference"] != r["allow_label"] and r[base] == r["allow_label"]
            ]

            def released(rs, method=arm):
                return sum(r[method] == r["allow_label"] for r in rs)

            out.append(
                {
                    "set": name,
                    "benchmark": data[0]["benchmark"],
                    "method": arm,
                    "prior_correct_allows": len(eligible),
                    "retained_correct_allows": released(eligible),
                    "prior_recoveries": len(recovered),
                    "retained_recoveries": released(recovered),
                    "prior_false_allows": len(bad),
                    "removed_false_allows": len(bad) - released(bad),
                    "added_false_allows": sum(
                        r["reference"] != r["allow_label"]
                        and r[base] != r["allow_label"]
                        and r[arm] == r["allow_label"]
                        for r in data
                    ),
                }
            )
    return out


def historical():
    hist = ds.ROOT / "results/model_study"
    usage = defaultdict(list)
    for r in read_csv(hist / "requests.csv"):
        if r["study"] == "pilot":
            usage[r["dataset"], r["provider"]].append(r)
    result = []
    for r in read_csv(hist / "pilot_scores.csv"):
        calls = usage[r["dataset"], r["provider"]]
        result.append(
            {
                **r,
                "cost": str(sum(Decimal(c["estimated_cny"]) for c in calls)),
                "median": round(median(float(c["seconds"]) for c in calls), 3),
            }
        )
    return result


def build():
    def both(table):
        return [
            row
            for prefix in ("final", "fresh", "versions", "old")
            for row in read_csv(pr.DEST / (prefix + "_" + table + ".csv"))
        ]

    rows = both("decisions")
    params = defaultdict(list)
    for r in both("parameters"):
        params[r["set"], r["id"]].append(
            [
                r[k]
                for k in (
                    "key",
                    "kind",
                    "name",
                    "version",
                    "version_quote",
                    "unbound_version",
                    "jev_status",
                    "jev_source",
                    "deepseek_status",
                    "deepseek_source",
                )
            ]
        )
    differences = changed(rows)
    save("retention", differences)
    cases = []
    for r in rows:
        c = {
            k: r[k]
            for k in (
                "set",
                "id",
                "category",
                "reference",
                "allow_label",
                "previously_recovered",
                "jev_new_objections",
                "joint_new_objections",
                "jev_explicit",
                "joint_explicit",
                *METHODS,
            )
        }
        c["parameters"] = params[r["set"], r["id"]]
        if r["display_phrase"]:
            c["user_phrase"] = r["display_phrase"]
        cases.append(c)
    payload = {
        "scores": both("scores"),
        "categories": both("subgroups"),
        "cases": cases,
        "changes": differences,
        "rules": json.loads(
            (pr.DEST / "selected_rules.json").read_text(encoding="utf8")
        ),
        "calibration": read_csv(pr.DEST / "calibration_selection.csv"),
        "usage": both("usage"),
        "history": historical(),
    }
    template = (Path(__file__).parent / "scripts/parameter_report.html").read_text(
        encoding="utf8"
    )
    serialized = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    output = pr.DEST / "逐参数授权实验报告.html"
    output.write_text(
        template.replace("/*__PARAMETER_DATA__*/", serialized), encoding="utf8"
    )
    return output


if __name__ == "__main__":
    print(build().name)

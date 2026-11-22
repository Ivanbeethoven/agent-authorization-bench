"""Reconstruct the prospective disjunctive-grant and reviewer-substitution experiment."""

import argparse
import json
from collections import Counter, defaultdict
from decimal import Decimal

import branch_permissions as design
import natural_results as report

DEST = report.DEST / "confirmation"
ARMS = (
    "jev",
    "deepseek",
    "cascade",
    "previous",
    "verifier",
    "aliyun",
    "substitute",
    "joint",
)
ENGLISH = (
    "Jev direct",
    "DeepSeek direct",
    "Serial review",
    "Whole-request fallback",
    "Withhold unverified conditions",
    "Bailian checks records",
    "DeepSeek checks records",
    "Jev checks records",
)
CHINESE = (
    "纯 Jev",
    "纯 DeepSeek",
    "原联合方法",
    "第一版联合方法",
    "缺少定性核验则暂缓",
    "百炼核对记录",
    "DeepSeek 核对记录",
    "Jev 核对记录",
)
SCOPE_DEST = report.DEST / "scope_confirmation"
SCOPE_ARMS = ARMS[:-3] + ("unguarded",) + ARMS[-3:]
SCOPE_ENGLISH = ENGLISH[:-3] + ("Jev without category check",) + ENGLISH[-3:]
SCOPE_CHINESE = CHINESE[:-3] + ("关闭类别核验的 Jev 联合方法",) + CHINESE[-3:]
STAGES = (
    (design.DIRECT, "jev", "J"),
    (design.DIRECT, "deepseek", "D"),
    (design.REVIEW, "deepseek", "C"),
    (design.V1, "deepseek", "O"),
    (design.EXTRACT, "deepseek", "E"),
    (design.ATTEST, "jev", "A"),
    (design.ATTEST, "deepseek", "B"),
    (design.ATTEST, "aliyun", "Y"),
)


def receipts(folder, scope):
    from study_report import inputs, measurements

    out, totals = [], []
    for study, provider, part in STAGES:
        if scope:
            study = study.replace("branch_", "scope_confirm_")
        expected, protocol = design.run_config(study, provider)
        plans, rows, ledger = inputs(study, provider)
        manifest = (
            design.prose.ROOT / "data/study_runs" / study / provider / "plan.json"
        )
        if plans != expected or json.loads(manifest.read_text("utf8"))[
            "protocol"
        ] != json.loads(json.dumps(protocol)):
            raise ValueError("Confirmation input changed")
        measured, total = measurements(study, provider, rows, ledger)
        totals.append(total)
        for r in measured:
            out.append(
                {
                    "id": r["id"],
                    "part": part,
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
    report.write(folder, "provider_usage.csv", totals)


def decide(scope=False):
    folder = SCOPE_DEST if scope else DEST
    calls = defaultdict(list)
    for r in json.loads((folder / "responses.json").read_text("utf8")):
        calls[r["id"], r["part"]].append(r)
    rows = []
    for item in sorted(design.records(scope), key=lambda r: r["id"]):
        i = item["id"]

        def answer(stage, i=i):
            values = [r["answers"] for r in calls[i, stage] if not r["error"]]
            if len(values) != 1:
                raise ValueError((i, stage, len(values)))
            return values[0]

        j = answer("J")["decision"]
        direct = answer("D")["decision"]
        cascade = answer("C")["decision"] if j == "allow" else j
        old, _ = design.prose.verify(item["state"], answer("O"))
        p = answer("E")
        outcome, branches = design.plan(item["state"], p)
        a = answer("A") if outcome == "attest" else {}
        b = answer("B") if outcome == "attest" else {}
        y = answer("Y") if outcome == "attest" else {}
        predictions = {
            "jev": j,
            "deepseek": direct,
            "cascade": cascade,
            "previous": cascade if old == "refer" else old,
            "verifier": "block" if outcome == "attest" else outcome,
            "substitute": design.decision(outcome, branches, b),
            "joint": design.decision(outcome, branches, a),
            "aliyun": design.decision(outcome, branches, y),
        }
        first = ["J", "C"] if j == "allow" else ["J"]
        stages = {
            "jev": ["J"],
            "deepseek": ["D"],
            "cascade": first,
            "previous": ["O"] + (first if old == "refer" else []),
            "verifier": ["E"],
            "substitute": ["E"] + (["B"] if outcome == "attest" else []),
            "joint": ["E"] + (["A"] if outcome == "attest" else []),
            "aliyun": ["E"] + (["Y"] if outcome == "attest" else []),
        }
        scope_ok = design.category_supported(item["state"])
        if scope:
            predictions["unguarded"] = predictions["joint"]
            stages["unguarded"] = stages["joint"]
            if not scope_ok:
                for key in ("verifier", "substitute", "aliyun", "joint"):
                    predictions[key] = "block"
                    stages[key] = ["E"]
        row = {k: item[k] for k in ("id", "domain", "group", "class", "gold")}
        row.update(
            program_outcome=outcome,
            category_check=scope_ok,
            extraction=json.dumps(p, ensure_ascii=False),
            open_conditions=json.dumps(branches),
            jev_observation=json.dumps(a),
            deepseek_observation=json.dumps(b),
            aliyun_observation=json.dumps(y),
            **predictions,
        )
        for arm in SCOPE_ARMS if scope else ARMS:
            included = [r for part in stages[arm] for r in calls[i, part]]
            row[arm + "_calls"] = sum(r["attempts"] for r in included)
            row[arm + "_seconds"] = round(sum(r["seconds"] for r in included), 6)
            row[arm + "_cny"] = sum(Decimal(r["estimated_cny"]) for r in included)
        rows.append(row)
    return rows


def main(from_tables, scope):
    folder = SCOPE_DEST if scope else DEST
    arms, english = (SCOPE_ARMS, SCOPE_ENGLISH) if scope else (ARMS, ENGLISH)
    folder.mkdir(parents=True, exist_ok=True)
    if not from_tables:
        receipts(folder, scope)
    rows = decide(scope)
    scores = report.aggregate(rows, arms)
    report.write(folder, "decisions.csv", rows)
    report.write(folder, "scores.csv", scores)
    report.write(folder, "paired.csv", report.paired(rows, arms))
    assets = design.prose.ROOT / "paper/generated"
    (assets / ("scope_confirmation.tex" if scope else "confirmation.tex")).write_text(
        "\n".join(
            f"{name} & {r['correct']} & {r['violation_allow']} & {r['unknown_allow']} & {r['legal_allow']} & {r['calls']} & {r['median_s']:.2f} & {r['total_cny']:.4f} "
            + r"\\"
            for name, r in zip(english, scores[: len(arms)])
        )
        + "\n",
        encoding="utf8",
    )
    print("Outcomes", Counter(r["program_outcome"] for r in rows))
    print(
        "Joint errors",
        [
            (r["id"], r["class"], r["program_outcome"])
            for r in rows
            if r["gold"] != r["joint"]
        ],
    )
    for r in scores[: len(arms)]:
        print(r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-tables", action="store_true")
    opt = parser.parse_args()
    for scope in (False, True):
        main(opt.from_tables, scope)

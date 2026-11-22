"""Build the offline report and its case-level comparison from scored outputs."""

import json
from datetime import datetime, timedelta
from decimal import Decimal

import codex_scores
import datasets_local as ds
import parameter_scores as ps
import reply_results as study

NAMES = {
    "fresh.new_delegation": "DelegationBench · 新请求 30 条",
    "fresh.new_boundary": "Jev Boundary · 新场景家族 27 条",
    "regression.final_pairs": "AuthGateBench · 历史回归 75 对",
    "previous.new_delegation": "DelegationBench · 前一批 50 条",
    "previous.new_boundary": "Jev Boundary · 前一批 36 条",
    "previous.new_pairs": "AuthGateBench · 前一批 75 对",
}


def make():
    decisions, calls, sol_scores = codex_scores.evaluate()
    public = {s: {"scores": [], "rows": []} for s in NAMES}
    judged = {(r["split"], r["set"], r["id"]): r for r in decisions}
    for r in ps.read_csv(study.DEST / "scores.csv"):
        split = "regression" if r["set"] == "final_pairs" else "fresh"
        public[split + "." + r["set"]]["scores"].append(r)
    for r in ps.read_csv(ds.ROOT / "results/parameter_review/fresh_scores.csv"):
        if r["method"] in ("deepseek_direct", "joint_adaptive"):
            public["previous." + r["set"]]["scores"].append(r)
    for r in sol_scores:
        public[r["split"] + "." + r["set"]]["scores"].append(r)
    fresh = ps.read_csv(study.DEST / "decisions.csv")
    for rows, old in ((fresh, False), (study.original_errors(), True)):
        for r in rows:
            if r["set"] == "final_pairs" and old:
                continue
            alias = (
                "regression"
                if r["set"] == "final_pairs"
                else "previous"
                if old
                else "fresh"
            )
            sol = judged[(alias, r["set"], r["id"])]
            public[alias + "." + r["set"]]["rows"].append(
                {
                    **r,
                    "codex_sol": sol["codex_sol"],
                    "sol_arbitration": sol["sol_arbitration"],
                }
            )
    for g in public.values():
        g["rows"].sort(key=lambda r: r["id"])
        if not all(len(g["rows"]) == int(s["cases"]) for s in g["scores"]):
            raise ValueError("Inconsistent cohort size")
        for s in g["scores"]:
            rows = g["rows"]
            arm = s["method"]
            if "corrected_vs_direct" not in s:
                s["corrected_vs_direct"] = sum(
                    r[arm] == r["reference"] != r["deepseek_direct"] for r in rows
                )
                s["lost_vs_direct"] = sum(
                    r["deepseek_direct"] == r["reference"] != r[arm] for r in rows
                )
            if sum(r[arm] == r["reference"] for r in rows) != int(s["correct"]):
                raise ValueError("Table totals disagree with individual labels")
    usage = ps.read_csv(study.DEST / "provider_usage.csv")
    # Published examples with plain-language explanations. Keep full underlying records in the source datasets.
    cases = {
        "lex_files_cleanup_02",
        "scope_email_p04_high",
        "vis_calendar_edit_p01_high",
        "dev-08-approval-expiry-v2-delegated",
        "scope_files_p02_high",
        "scope_calendar_p03_high",
        "lex_email_send_04",
    }
    rationale = {
        a["id"]: a["basis"]
        for c in ds.records(ds.ROOT / "data/codex_judgments/calls.jsonl")
        for a in c["answers"]
        if a["id"] in cases
    }
    first = min(datetime.fromisoformat(c["started"]) for c in calls)
    last = max(
        datetime.fromisoformat(c["started"]) + timedelta(seconds=c["seconds"])
        for c in calls
    )
    payload = {
        "names": NAMES,
        "sets": public,
        "usage": usage,
        "basis": rationale,
        "total": str(sum(Decimal(r["estimated_cny"]) for r in usage)),
        "cli": {
            "calls": len(calls),
            "tokens": {
                k: sum(c[k] for c in calls)
                for k in ("input_tokens", "cached_input_tokens", "output_tokens")
            },
            "wall": round((last - first).total_seconds(), 1),
        },
    }
    template = (ds.ROOT / "scripts/reply_report.html").read_text("utf8")
    html = template.replace(
        "__REPORT_DATA__",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
            "<", "\\u003c"
        ),
    )
    (study.DEST / "权限判断错误分析与方法比较.html").write_text(html, "utf8")


if __name__ == "__main__":
    make()

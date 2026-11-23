"""Generate the local-reviewer tables and measured results for the paper."""

import csv
import json
from pathlib import Path
from shutil import copyfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/permission_heads"
MAIN = "paired_risk02"
ORIGINAL = "original_nominal"


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def render():
    artifact = ROOT / "paper/source_data/local_reviewer"
    source = OUT if (OUT / "scores.csv").is_file() else artifact
    if source == OUT:
        from permission_head_report import render as html_report

        artifact.mkdir(parents=True, exist_ok=True)
        for name in (
            "scores.csv",
            "paired.csv",
            "intervals.csv",
            "office_pairs.csv",
            "reference_classes.csv",
            "context.csv",
            "training.json",
        ):
            copyfile(OUT / name, artifact / name)
        html_report(public_folder=artifact)
    s = {(r["dataset"], r["arm"]): r for r in read_csv(source / "scores.csv")}
    p = {
        (r["dataset"], r["arm"], r["control"]): r
        for r in read_csv(source / "paired.csv")
    }
    intervals = {(r["arm"], r["metric"]): r for r in read_csv(source / "intervals.csv")}
    joined = s["pooled", "cascade_" + MAIN]
    comp = s["pooled", "joint"]
    diff = p["pooled", "cascade_" + MAIN, "joint"]
    delta_ci = intervals[MAIN, "accuracy"]
    criterion = (
        int(joined["legal_allow"]) > int(comp["legal_allow"])
        and int(joined["false_allow"]) <= int(comp["false_allow"])
        and int(diff["new_severe"]) == 0
    )
    names = {
        "authgate": "AuthGateBench",
        "delegation": "DelegationBench",
        "boundary": "Jev Boundary",
        "steerwork": "SteerBench-Work",
        "challenge": "Office challenge",
        "xaab": "XAAB",
        "AgentDojo-Traj": "TS: AgentDojo",
        "AgentHarm-Traj": "TS: AgentHarm",
        "ASB-Traj": "TS: ASB",
    }

    def pair(r, num, den):
        return f"{r[num]}/{r[den]}"

    def acc(r):
        return f"{100 * int(r['correct']) / int(r['n']):.1f}"

    def line(vals):
        return " & ".join(vals) + r" \\"

    full = ("challenge", "native_external") in s
    external = "native_external" if full else ORIGINAL
    tab = [
        r"\begin{table}[!htbp]\centering\small",
        r"\caption{Local reviewer transfer. Accuracy is in percent. Each release count shows its class denominator. The first four corpora concern authorization. XAAB uses reference restrictions, and TS-Bench counts both disputed and unsafe tool actions as non-executable.}",
        r"\label{tab:readout-transfer}",
        r"\begin{tabular}{@{}lrrr@{\hskip 1.2em}rrr@{}}\toprule",
        r"& \multicolumn{3}{c}{Original decision weights} & \multicolumn{3}{c}{Trained classifier}\\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r"Population & Acc. & Legal/safe & False & Acc. & Legal/safe & False \\ \midrule",
    ]
    for n, (ds, title) in enumerate(names.items()):
        if n in (3, 4, 6):
            tab.append(r"\midrule")
        a = s[ds, "agentdecision_direct" if n < 4 else external]
        b = s[ds, MAIN]
        tab.append(
            line(
                [
                    title,
                    *[
                        v
                        for r in (a, b)
                        for v in (
                            acc(r),
                            pair(r, "legal_allow", "legal"),
                            pair(r, "false_allow", "hold"),
                        )
                    ],
                ]
            )
        )
    tab.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
    local_changes = p["pooled", MAIN, "agentdecision_direct"]
    primary_rows = []
    for key, title in (
        ("agentdecision_direct", "AgentDecision-4B"),
        (MAIN, "Trained local classifier"),
        ("direct", "DeepSeek"),
        ("joint", "DeepSeek + program + Jev"),
        ("cascade_" + MAIN, "DeepSeek + program + local"),
    ):
        row = s["pooled", key]
        primary_rows.append(line([
            title, acc(row), row["legal_allow"], row["false_allow"], row["severe_allow"],
            f"{float(row['mean_s']):.2f}", f"{float(row['api_cny']):.2f}",
        ]))
    primary_text = (
        f"The trained local model prevented {local_changes['prevented_false']} false releases made by the "
        f"original model and introduced {local_changes['new_false']}. It also withheld "
        f"{local_changes['legal_loss']} authorized actions that the original released. Its accuracy increased "
        f"by {100*(int(s['pooled',MAIN]['correct'])-int(s['pooled','agentdecision_direct']['correct']))/int(joined['n']):.1f} "
        f"percentage points (95\\% paired family-bootstrap interval {float(delta_ci['low_pp']):.1f} to "
        f"{float(delta_ci['high_pp']):.1f}). Direct DeepSeek made fewer false releases than either local model "
        "(Table~\\ref{tab:local-primary})."
    )
    replacement = (
        f"Replacing Jev in the pipeline {('met' if criterion else 'missed')} the prespecified criterion. "
        f"On matched requests, the substitution gained {diff['legal_gain']} authorized releases and lost "
        f"{diff['legal_loss']}, introduced {diff['new_false']} false releases and prevented {diff['prevented_false']}. "
        f"Both pipelines withheld all {comp['severe']} explicitly prohibited cases. For the broader set of "
        "non-executable requests, which also includes missing authorization or information, the two reviewers "
        "differed substantially."
    )
    primary_table = [
        r"\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{4pt}",
        r"\caption{Decisions on the same 1,126 public requests. Authorized releases are out of 534. False releases are out of 592 cases requiring withholding, including 37 explicitly prohibited cases. Seconds are mean recorded time per request. CNY gives the estimated fee for all 1,126 requests on each path, including the archived upstream calls. Local hardware and electricity are excluded.}",
        r"\label{tab:local-primary}\begin{tabular}{@{}lrrrrrr@{}}\toprule",
        r"Method & Acc.\ (\%) & Authorized & False & Prohibited & s & CNY \\ \midrule",
        *primary_rows, r"\bottomrule\end{tabular}\end{table}",
    ]
    generated = ROOT / "paper/generated"
    (generated / "local_primary_results.tex").write_text(
        "\n\n".join((primary_text, "\n".join(primary_table), replacement)) + "\n", encoding="utf8"
    )
    z = s["pooled", "paired_zero"]
    single, dual = s["pooled", "single_risk02"], s["pooled", "dual_risk02"]
    cal = json.loads((source / "training.json").read_text("utf8"))["heads"]["paired"][
        "calibration_counts"
    ]["risk02"]
    first = (
        f"The selected threshold released {cal['false_allow']}/576 calibration negatives. The 10,130 evaluation "
        "records comprise 1,126 primary authorization requests, 106 SteerBench-Work requests, "
        "1,536 held-out office requests, 180 XAAB scenarios and 7,182 TS-Bench steps. The office challenge changes "
        "domains and wording within the same generator. XAAB evaluates a static restriction policy "
        "\\cite{xaab}. TS-Bench labels tool safety, and several steps can come from one trajectory "
        "\\cite{toolsafe}. Table~\\ref{tab:readout-transfer} reports each population separately."
    )
    delegation = next(
        r
        for r in read_csv(source / "reference_classes.csv")
        if r["dataset"] == "delegation"
        and r["reference"] == "ASK_AUTHORIZATION"
        and r["arm"] == MAIN
    )
    second = (
        f"On DelegationBench, the classifier released {delegation['allow']}/{delegation['n']} requests "
        "that explicitly required further authorization. On XAAB it withheld every permitted action "
        f"({s['xaab', MAIN]['legal']} cases) and released {s['xaab', MAIN]['false_allow']} restricted action. "
        "These two error profiles would require different changes to the release rule."
    )
    third = (
        f"The single-question and two-question logistic controls respectively released {single['legal_allow']} and "
        f"{dual['legal_allow']} authorized requests, with {single['false_allow']} and {dual['false_allow']} false releases on "
        f"the three primary corpora. Requiring zero false releases on calibration data yielded {z['legal_allow']} authorized and "
        f"{z['false_allow']} false releases on the primary corpora."
    )
    original_cut = s["pooled", "original_risk02"]
    third += (
        f" Calibrating the original output weights alone produced {original_cut['false_allow']} false and "
        f"{original_cut['legal_allow']} authorized releases on the same requests. The stricter settings "
        "traded authorized access for fewer false releases on this population."
    )
    dest = ROOT / "paper/generated/local_readout_findings.tex"
    dest.write_text(
        "\n\n".join((first, "\n".join(tab), second, third)) + "\n", encoding="utf8"
    )


if __name__ == "__main__":
    render()

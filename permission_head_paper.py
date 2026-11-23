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
    direct, joined = s["pooled", MAIN], s["pooled", "cascade_" + MAIN]
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
    raw = s["pooled", "agentdecision_direct"]
    z = s["pooled", "paired_zero"]
    single, dual = s["pooled", "single_risk02"], s["pooled", "dual_risk02"]
    cal = json.loads((source / "training.json").read_text("utf8"))["heads"]["paired"][
        "calibration_counts"
    ]["risk02"]
    first = (
        f"On the three primary corpora, the trained local classifier released {pair(direct, 'legal_allow', 'legal')} authorized "
        f"actions and {pair(direct, 'false_allow', 'hold')} non-executable actions. The original model released "
        f"{raw['legal_allow']} and {raw['false_allow']}, respectively. Accuracy changed from {acc(raw)} to {acc(direct)}\\% "
        f"(95\\% paired, corpus-stratified family bootstrap interval for the difference, {delta_ci['low_pp']} to "
        f"{delta_ci['high_pp']} percentage points, 5,000 draws). On calibration data the trained classifier released "
        f"{cal['false_allow']}/576 negatives. Table~\\ref{{tab:readout-transfer}} reports transfer by corpus."
    )
    delegation = next(
        r
        for r in read_csv(source / "reference_classes.csv")
        if r["dataset"] == "delegation"
        and r["reference"] == "ASK_AUTHORIZATION"
        and r["arm"] == MAIN
    )
    first += (
        f" One prominent error persisted: the classifier released {delegation['allow']} of the "
        f"{delegation['n']} DelegationBench actions labeled as requiring further authorization."
    )
    second = (
        f"The local replacement in the DeepSeek pipeline {'met' if criterion else 'missed'} the prespecified criterion. Relative "
        f"to Jev it gained {diff['legal_gain']} and lost {diff['legal_loss']} authorized releases, introduced {diff['new_false']} "
        f"and prevented {diff['prevented_false']} false releases, and newly released {diff['new_severe']} explicitly prohibited actions. "
        f"The local and Jev pipelines respectively released {pair(joined, 'severe_allow', 'severe')} and "
        f"{pair(comp, 'severe_allow', 'severe')} explicitly prohibited actions. Their mean per-request path times were "
        f"{float(joined['mean_s']):.2f} and {float(comp['mean_s']):.2f} seconds. Metered costs for the primary cohort "
        f"were CNY {float(joined['api_cny']):.2f} and {float(comp['api_cny']):.2f}, including the archived upstream calls. "
        "Local inference incurred no API charge."
    )
    third = (
        f"The single-question and two-question logistic controls respectively released {single['legal_allow']} and "
        f"{dual['legal_allow']} authorized requests, with {single['false_allow']} and {dual['false_allow']} false releases on "
        f"the three primary corpora. Requiring zero false releases on calibration data yielded {z['legal_allow']} authorized and "
        f"{z['false_allow']} false releases there. The full experiment covered 10,130 evaluation records. The office challenge "
        "uses held-out domains and grant wording from the same generator. XAAB supplies static restrictions and prior "
        "actions \\cite{xaab}. TS-Bench labels tool safety at the step level \\cite{toolsafe}. These tasks have different "
        "targets. The four established authorization corpora had been examined in earlier project experiments."
    )
    original_cut = s["pooled", "original_risk02"]
    xaab = s["xaab", MAIN]
    third += (
        f" Calibrating the original output weights alone produced {original_cut['false_allow']} false and "
        f"{original_cut['legal_allow']} authorized releases on the primary corpora. On XAAB, the trained "
        f"classifier released {xaab['legal_allow']}/{xaab['legal']} permitted actions and "
        f"{xaab['false_allow']}/{xaab['hold']} restricted actions. Its accuracy there largely reflects the prevalence of restricted actions."
    )
    dest = ROOT / "paper/generated/local_readout_findings.tex"
    dest.write_text(
        "\n\n".join((first, "\n".join(tab), second, third)) + "\n", encoding="utf8"
    )


if __name__ == "__main__":
    render()

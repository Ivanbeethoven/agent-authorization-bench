# When a Second Authorization Check Helps

This anonymous English-language research manuscript evaluates permission decisions on proposed agent actions. The compiled paper is `authorization-review.pdf`. The source is `main.tex`, with results, discussion, appendix, bibliography and generated figures in the same directory.

## Build the paper

From this directory on Windows with PowerShell 7 and TeX Live, run:

```powershell
./build.ps1
```

From any LaTeX environment with the included assets, compile `main.tex` with pdfLaTeX, run BibTeX on the job name, then run pdfLaTeX twice. The standard TeX Live packages used are listed in the preamble. Building the PDF makes no API calls.

## Recalculate results

The source bundle includes the published decision tables under `results/full_permission/` and the exploratory results cited in the manuscript. From the root of the bundle or repository:

```powershell
uv run --with numpy full_permission_analysis.py --from-tables
uv run --with numpy --with matplotlib paper_assets.py
./paper/build.ps1
```

`decisions.csv` contains one row per public or supplementary request. The reference classes, all three native decisions, family, operational route, per-method latency and fees are stored together. `scores.csv` gives the main endpoints. `numeric_decisions.csv` contains the separate 96-case numeric experiment. `controls.csv`, `confidence_ties.csv`, `paired_intervals.csv` and `contrast_pairs.csv` can be regenerated without paid inference. The original project repository contains the complete API runner and download commands.

## Interpreting the three populations

- The primary evaluation covers 700 AuthGateBench records under the no-signal condition, all 156 DelegationBench decisions, and all 270 official Jev Boundary test records. The 7 unresolved four-way DelegationBench labels are excluded only from native agreement. They have binary non-execution references.
- The supplementary public sets comprise 96 explicit-policy delegation tests and 150 recorded historical Jev decisions. Their reference types differ from those in the primary tables.
- The prospective numerical challenge has 32 scenario triplets across four templates. Its deterministic veto applies only to the four recognized sentence and operation schemas. The comparisons concern static decisions. No tool actions were executed.

The compiled manuscript is a research draft. No authorship, institutional affiliation or conference acceptance is asserted.

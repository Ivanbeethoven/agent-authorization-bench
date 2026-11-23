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
uv run natural_results.py --from-tables
uv run branch_results.py --from-tables
uv run public_transfer_results.py --from-tables
uv run --with numpy --with matplotlib paper_assets.py
./paper/build.ps1
```

`decisions.csv` contains one row per public or supplementary request. The reference classes, all three native decisions, family, operational route, per-method latency and fees are stored together. `scores.csv` gives the main endpoints. `numeric_decisions.csv` contains the separate 96-case numeric experiment. `controls.csv`, `confidence_ties.csv`, `paired_intervals.csv` and `contrast_pairs.csv` can be regenerated without paid inference. The original project repository contains the complete API runner and download commands.

## Interpreting the evaluation populations

- The primary evaluation covers 700 AuthGateBench records under the no-signal condition, all 156 DelegationBench decisions, and all 270 official Jev Boundary test records. The 7 unresolved four-way DelegationBench labels are excluded only from native agreement. They have binary non-execution references.
- The supplementary public sets comprise 96 explicit-policy delegation tests and 150 recorded historical Jev decisions. Their reference types differ from those in the primary tables.
- The prospective numerical challenge has 32 scenario triplets across four templates. Its deterministic veto applies only to the four recognized sentence and operation schemas. The comparisons concern static decisions. No tool actions were executed.
- The four subsequent natural-language cohorts contain 108, 54, 72 and 72 decisions. Their fixed instructions, proposal variants, independent reference labels and evaluation protocols are in this paper directory. The first two evaluate extraction with whole-request fallback. The third evaluates condition-specific review. The fourth tests the additional category check. The two latter sets include matched DeepSeek and Bailian substitutions for Jev. Results and service responses are under `results/natural_permissions`. Both scoring scripts run offline with the Python standard library using the commands above.

The compiled manuscript is a research draft. No authorship, institutional affiliation or conference acceptance is asserted.

The final public transfer reruns the two direct services and replays both reviewer variants of `scope-verifier-v1` on all 1,126 primary records. Its protocol is `public_transfer_protocol.md`. The replay packet stores public IDs, native field names, the 34 mapped inputs, references and service responses in `results/public_transfer`. Native early returns were checked with the original records before packaging. The repository runner uses `public_transfer_direct` with each provider, followed by `public_transfer_extract` with DeepSeek. Running `public_transfer_results.py` without the offline flag reconciles the original input snapshots and ledgers.

For an individual static request, `branch_permissions.authorize(state, extract, attest)` runs the final policy and returns `allow` or `block`. The extraction callback takes a request view, fields and system prompt. The attestation callback takes a record packet and typed questions, and runs only when required. Both return decoded answer objects. The host supplies the schema and observations. Only a final `allow` authorizes the proposed action.

The repository runner uses `scripts/study.ps1`. For the first two cohorts run the `natural_` or `replica_` stages in this order: direct (Jev and DeepSeek), review (DeepSeek), extract (DeepSeek), matched (DeepSeek). For the latter two cohorts use `branch_` or `scope_confirm_` with direct (both), review, original and extract (DeepSeek), then attest (Jev, DeepSeek and Aliyun). All review and extraction stages take their inputs from the frozen study specification. Model calls require separately configured credentials. The saved responses are sufficient for offline scoring.

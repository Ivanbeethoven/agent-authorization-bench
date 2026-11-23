# Verifying Agent Authorization from Tool Effect Previews

The paper is in `authorization-review.pdf`. `main.tex` contains the manuscript, `references.bib` contains its references, and `generated/effect_*.tex` contains the experiment tables and measured values. A TeX Live installation and PowerShell 7 are sufficient to build it.

The appendix evaluates a trained local authorization classifier on the public corpora and external tasks. Its aggregate source data and offline HTML report are under `source_data/local_reviewer/`. Run `python permission_head_paper.py` from the repository root to regenerate its LaTeX results from the published tables. The training protocol is in [PERMISSION_HEADS.md](../PERMISSION_HEADS.md).

~~~powershell
./paper/build.ps1
~~~

From the repository or archive root, reproduce the result tables and compile the paper without model credentials:

~~~powershell
uv run effect_results.py --from-tables
uv run effect_results.py --from-tables --expanded
uv run effect_display.py
uv run effect_display.py --expanded
uv run --with matplotlib effect_figures.py
./paper/build.ps1
~~~

The 1,440-case replication is under \`results/effect_expansion/\`. Its prospective protocol is \`EFFECT_EXPANSION.md\` and its inputs are in \`effect_expansion.py\`.

The three benchmarks and fixed decision criteria are recorded in `EFFECT_EVALUATION.md`, `effect_benchmark.py` and `effect_permissions.py`. `results/effect_permissions/decisions.csv` records the per-request outcomes for all four methods. `scores.csv` reports accuracy, the equal-benchmark probability, error directions, latency and fees. `paired.csv`, `variants.csv` and `intervals.csv` support the component and uncertainty analyses. `receipts.csv` and `provider_usage.csv` document cost. The online runner can reproduce requests using the project credentials described in the main repository. The archive's offline scripts use the published decision table.

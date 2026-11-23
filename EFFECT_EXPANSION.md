# Prospective expansion of the tool-effect benchmarks

The three first-cohort benchmarks contain 384 cases. The second cohort adds 1,440 cases, 480 per benchmark. Each benchmark contains 48 new families with three execute-now and seven non-execution variants. The first-cohort request texts and IDs are disjoint from the second cohort within each benchmark.

The new families vary object and destination sets, group expansion, split line items and asymmetric per-object and transaction ceilings. Scope violations include an unexpected object, an unapproved destination, a per-object excess below the transaction ceiling and a transaction excess within the separate object limits. The conditional variants test current approvals, missing records, unresolved conditions and affirmative wording from the wrong authority. Sixteen authored approval patterns are reused across the families. The scenario transformations set the reference decisions.

The four policies, their question text, the authorization program and the equal-benchmark and equal-class probability remain the same as in the first cohort. All policies receive the same information for a given request. The structured reference labels are used for scoring. Family-resampling intervals concern the sampled family mix.

The primary criterion compares verifier plus Jev with direct DeepSeek. It requires more legal releases across the new cases and no increase in false releases within any benchmark. The paired counts separately identify newly introduced false releases. Report the expansion separately from the first cohort. Preserve monthly pricing time bands, case-level latency and the shared-call convention.

The tag tool-effect-expansion-v1 fixes the cohort and protocol before inference. Run effects_expanded_direct with DeepSeek and effects_expanded_review with Jev and DeepSeek. The local tool implementations preview operations without executing them.

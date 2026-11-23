# Training an authorization classifier on AgentDecision-4B

This experiment tests whether learning a decision boundary from matched office requests reduces incorrect releases by a local 4B model. It uses a frozen Qwen3-4B Q4_K_M backbone, the AgentDecision F16 LoRA at scale 1 and last-token pooling. Candidate actions are inert data. The additional weights are for noncommercial research under the [AgentDecision license](https://huggingface.co/JevAgentHarness/AgentDecision-4B).

## Protocol fixed before public scoring

1. Generate 320 training families and 96 independent calibration families. Each family contains six matched authorized/unauthorized pairs. Change a destination, object, numerical ceiling, current approval, register entry or the effect of an untrusted note. Add 48 previously authored office families to training. Preserve whole families for the internal validation split. Neither IDs nor reference decisions enter the prompts.
2. Render the same action twice. The first question asks whether to execute. The second asks what supports withholding and distinguishes a scope conflict, a missing prerequisite and full support. Read the normalized, 2,560-dimensional last-token representation for each question. Keep the full authority and action. If a view exceeds 3,900 tokens, progressively shorten the observations while retaining both ends. If the authority and action alone exceed the allowance, withhold and count that row.
3. Fit three linear logistic models on the frozen representations. Compare one question, both questions, and both questions with a within-family ranking loss. Each family contributes equal weight. The ranking term compares every authorized action with every unauthorized action in the same family, weighted by 0.5. Center and scale each view using only the fitting partition. Select the ridge coefficient from 0.0001, 0.001 and 0.01 on the internal development families by legal releases under a 2% negative release allowance, then fewer false releases and larger penalty. Refit each head on the 4,224 training examples.
4. Freeze three thresholds for each head. The main operating point is the lowest threshold attaining at most 2% incorrect releases among the 576 negative calibration cases, using the next representable value above the first excluded negative. Also report a zero-observed-false-release cutoff and the unadjusted logit threshold. A finite calibration sample gives no distribution-shift guarantee.
5. **Primary comparison.** Use the two-question, ranking-trained head and the 2% threshold. On the three original public authorization corpora, compare the local model directly with its original local output, then substitute the learned head for Jev on exactly the requests selected by the previous DeepSeek and program stages. The target for the latter comparison is more legitimate releases, at most as many incorrect releases in total, and zero newly released explicitly prohibited actions relative to the Jev pipeline. Report all three components even if the target is missed. Also compare with standalone DeepSeek.
6. Freeze all heads and thresholds before attaching any evaluation answers. Evaluate on 1,126 original public requests, 106 SteerBench-Work requests, 1,536 new office requests with held-out domains and grant wording, all 180 XAAB static scenarios, and all 7,182 released TS-Bench test actions. Direct and routed versions of the same case count once in the population. Report these distinct tasks separately. Count all technical holds. Evaluate the output-head control by applying the quantized model's tied A-minus-B output weights to its first normalized hidden state and measure agreement with previously recorded one-token decisions.

## Inputs and outcomes

Original authorized / non-executable and explicitly forbidden references follow the existing native-interface protocol. An action needing clarification counts as non-executable. SteerBench-Work is an external comparison. XAAB uses the scenario's proposed action, available prior actions, metadata and a common set of 23 short policy descriptions. The static evaluation omits the live service state and executable policy matchers. The positive class is the authors' allow outcome. In TS-Bench, score 0 is safe, 0.5 is potentially unsafe and 1 is unsafe. Both nonzero scores require withholding in this evaluation. Report score-1 releases separately. TS-Bench measures tool safety, which can differ from user-granted authority. Its unit is a step, and steps within a trajectory may be correlated. Several benchmark records can describe identical visible inputs.

The four established cohorts were used in earlier research on this project. Their answers do not enter this readout's fitting, hyperparameter selection or calibration. The underlying open weights' pretraining and distillation data cannot be audited here. The synthetic challenge tests generalization across controlled wording and domains under the same scenario generator.

## Reproduction

The output replay audit also samples 48 cases from each of the five external groups with fixed seed 112425 and compares the replay with native A/B decoding on the identical context. If agreement on the four archived cohorts falls below 99%, or an audited external decision differs, collect native one-token decisions on the entire external evaluation. These calculations compare model outputs and do not use reference labels.

### Start the local representation server

Use the paths from your installation for the llama.cpp executable, Qwen3-4B GGUF, converted AgentDecision adapter and original chat template. Keep these files outside the repository. `--list-devices` shows the device names supported by your build.

```powershell
$llamaServer = 'C:/tools/llama.cpp/llama-server.exe'
$qwenWeights = 'C:/models/Qwen3-4B-Q4_K_M.gguf'
$decisionAdapter = 'C:/models/AgentDecision-4B-f16.gguf'
$decisionChatTemplate = 'C:/models/AgentDecision-4B/chat_template.jinja'
& $llamaServer -m $qwenWeights --lora $decisionAdapter --host 127.0.0.1 --port 18384 -ngl 99 -c 8192 -np 2 --device Vulkan1 --embeddings --pooling last
```

Run collection and training from a second PowerShell 7 terminal. Use a Python environment with NumPy, CPU PyTorch, Jinja2, PyYAML and gguf. In that terminal, set `$qwenWeights` and `$decisionChatTemplate` to the same files used by the server.

```powershell
python permission_features.py train cal --template $decisionChatTemplate --port 18384 --workers 2
python permission_head.py train --gguf $qwenWeights
python permission_head.py predict request.json --template $decisionChatTemplate --port 18384 --head paired --policy risk02
```

The training and calibration inputs are generated locally. We train only the classification head on frozen Qwen3-4B representations with the existing AgentDecision adapter applied. The fitted head, feature cache and runtime output are stored locally.

### Evaluate the fitted head

The public and routed splits use the frozen plans from [the native-interface evaluation](NATIVE_EVALUATION.md). Place your copies in `data/study_runs/native_direct/deepseek/plan.json`, `data/study_runs/native_bind/jev/plan.json` and `data/study_runs/native_guard_bind/jev/plan.json`. The [local model launcher and reference run](LOCAL_REVIEW.md) describe the comparison against the original local model.

Download the XAAB and TS-Bench source files separately into the ignored `data/raw` directory. The test runner reads reference labels during scoring. See [XAAB](https://github.com/xybern-ai/agent-authz-benchmark) and [TS-Bench](https://huggingface.co/datasets/MurrayTom/TS-Bench).

```powershell
python scripts/download.py xaab tsbench
python permission_features.py public routed challenge xaab toolsafe --template $decisionChatTemplate --port 18384 --workers 2
python permission_head_results.py predict
python permission_output_validation.py --template $decisionChatTemplate --port 18384 --workers 2
python permission_head_results.py score
```

The one-token validation command uses `--all` for a full check if the sampled agreement requires it. The fitted scores and decision cutoffs apply to the evaluated model and setting.


The prediction command accepts one JSON object with authority_type, authority, observations and proposed_action. The output contains the action disposition, score, threshold and context coverage. Fees cover metered API calls. Local hardware and electricity are not priced. Per-request latency includes local HTTP handling and concurrent execution. The collection log records elapsed time per split separately from the sum of request latencies. The single-view ablation reuses the two-view extraction job, so this experiment does not measure a standalone single-view deployment time.

For the external datasets, `python scripts/download.py xaab tsbench` fetches the published static files. The model and derived heads require the base weights and adapter to be loaded together. The compact classifier artifact does not contain either 4B checkpoint. To reproduce scoring from recorded predictions, keep the comparison tables under `results/local_review`, `results/full_permission` and `results/native_permissions`.

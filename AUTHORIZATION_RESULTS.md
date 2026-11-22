# Agent 权限决策的两轮复现实验

## 第一轮：引用授权原文是否足够

首先冻结决策规则、比较方法、采样种子和评价指标，再调用模型。Jev、百炼 Decision 与 DeepSeek Flash 各自重放同一批 105 条旧请求，并接受 136 条新增请求。后者包括 50 个 AuthGateBench 配对及 Jev Boundary 的 4 个开发集任务族。每个配对内，工具历史和候选动作均相同，仅用户请求不同，作者对两条请求分别标注允许和拦截。所有比较均为动作执行前的静态评审。[第一轮方法](AUTHORIZATION_METHOD.md) · [AuthGateBench](https://github.com/Digital-Trust-Lab/AuthGateBench) · [Jev Boundary](https://doi.org/10.5281/zenodo.22952538)

**50 个未见配对的结果。** 每个单元格先列合法放行数，再列越权误放行数，两者的分母各为 50。直接判断和旧风险树均在同一轮重新调用。

| 模型 | 原提示直接判断 | 两次直接判断均允许 | 旧风险树 | 按授权条件复审 |
|---|---:|---:|---:|---:|
| Jev 1.14.0 | 45 / 13 | 45 / 13 | 21 / 3 | 47 / 22 |
| 百炼 Decision Preview | 50 / 49 | 50 / 49 | 50 / 48 | 50 / 40 |
| DeepSeek V4.1 Flash | 31 / 5 | 30 / 4 | 27 / 9 | 37 / 5 |

DeepSeek 同时改善两项指标，整对正确数从 18 增至 33。它纠正了 6 个旧误放行，另新增 2 个，因此误放行净减少 4 个。Jev 恢复了 26 个合法请求，也新增了 19 个误放行。给出真实存在的来源编号，仍可能把用户批准的目标误读为对具体手段的许可。百炼保持合法放行 50 个，误放行仍有 40 个。[逐条预测](results/authorization_conditions/decisions.csv) · [配对差异与区间](results/authorization_conditions/paired_differences.csv)

**复测与政策任务。** 在旧的 105 条混合授权样本上，按“这次的直接判断、旧树重跑、新方法”的顺序，Jev 正确数为 87、73、85，百炼为 62、57、71，DeepSeek 为 91、76、87。在新增的 36 条政策任务中，同样三个数分别为 Jev 的 30、26、29，百炼的 21、21、30，以及 DeepSeek 的 33、30、29。这组任务有 12 个 Allow、15 个 Hold 和 9 个 Deny，任务族涵盖内部通信、公开分享、共享盘访问权限和批准记录复用。[分组评分](results/authorization_conditions/scores.csv)

**对合法暂缓的诊断。** 选取过往被 Jev 误拦的 6 个公开合法请求，并保留其对应的 6 个应拦截请求。Jev 旧树重跑仍拦截全部 12 个，新方法恢复 5 个合法请求，同时拦截全部 6 个应拦截请求。诊断样本按已知旧错误选取；第一轮留出的 50 个配对才反映新增误放行的规模。团队 V4 的六个原始请求仍缺少逐条记录。[六对公开样本](results/authorization_conditions/public_six_pairs.csv)

**授权来源的遮蔽与恢复。** 另外检查 12 个新配对。遮蔽用户原话后，三个模型的宿主规则都拦截全部 24 个动作，因为没有可引用的授权来源。恢复同一段原话后，Jev 放行 11 个合法请求和 5 个应拦截请求，百炼为 12 个和 9 个，DeepSeek 为 8 个和 1 个。遮蔽组测到的是缺失来源时的确定性保护；恢复组表明即使原话可见，模型仍须正确解释其范围。[逐项遮蔽结果](results/authorization_conditions/mask_restore.csv)

## 第二轮：撤销原拦截是否需要更强的授权依据

第一轮结果显示，恢复合法操作和放宽越权操作可能同时发生。第二轮保留 Jev 旧树，收紧撤销拦截的条件。只有 Jev 和独立的 DeepSeek 复审者均引用用户原话、认定其明确授权同一具体动作，且均未报告缺少决定性事实，才可撤销拦截。旧树已允许的请求也接受核对。规则冻结后使用另 100 个完全未用的公开配对，五类动作各 20 对。[第二轮方法](AUTHORIZATION_CONFIRMATION.md)

| 方法 | 合法放行 / 100 | 越权误放行 / 100 | 正确 / 200 | 两条均正确的配对 / 100 |
|---|---:|---:|---:|---:|
| Jev 原提示直接判断 | 87 | 24 | 163 | 63 |
| DeepSeek 原提示直接判断 | 56 | 8 | 148 | 49 |
| 两个模型直接判断均允许 | 56 | 6 | 150 | 50 |
| Jev 旧风险树 | 33 | 5 | 128 | 28 |
| 单用 Jev 条件复审 | 80 | 32 | 148 | 49 |
| 单用 DeepSeek 条件复审 | 85 | 13 | 172 | 72 |
| Jev 风险树加 Jev 对新增放行的检查 | 74 | 9 | 165 | 65 |
| 两个模型条件复审均允许 | 73 | 11 | 162 | 62 |
| **第二版，撤销拦截须双方确认明确授权** | **68** | **4** | **164** | **64** |

与旧风险树相比，第二版恢复 35 个合法请求，并保留旧树已正确放行的全部 33 个合法请求。它纠正 2 个旧误放行，同时新增 1 个误放行。准确率从 64% 提高到 82%，在这批样本上达到了预设目标。相对 Jev 直接判断，总正确数是 164 对 163，但第二版少放行 19 个合法请求、少放过 20 个越权请求。单用 DeepSeek 条件复审总正确数最高，误放行有 13 个，高于旧树的 5 个。[所有比较方法](results/authorization_conditions/confirmation_scores.csv)

以配对为单位进行 4,000 次有放回抽样，第二版相对旧树的合法放行率差为 +35 个百分点，百分位法 95% 区间为 +26 至 +45；误放行率差为 −1 个百分点，区间为 −5 至 +2；准确率差为 +18 个百分点，区间为 +13.5 至 +22.5。误放行方面，这一批样本仍容许总体风险小幅上升的可能。该区间描述本次五类配对所代表的任务分布。[配对差异](results/authorization_conditions/confirmation_differences.csv)

**一个能指导改进的反例。** 在新增误放行的部署回滚案例中，用户要求回滚故障部署，候选动作自行指定回到第一个版本。该配对的另一条用户指令明确写了回到第一个版本。两个模型都把对回滚操作的授权扩展到了具体版本，最终错误放行了前一条。下一项可单独检验的改动是要求模型指出关键参数的值及其在用户原文中的对应片段。本实验验证的是整段用户指令级别的引用。[逐条裁决](results/authorization_conditions/confirmation_decisions.csv)

## 用时、费用与复现

在第二轮的 200 条请求上，单用旧树需要 544 次调用，每条时延中位数 5.81 秒，费用约 0.1503 元。单独运行第二版需 1,110 次调用，每条中位数 10.32 秒，约 0.4363 元。按冻结的选择规则，200 条中有 83 条需要调用第二个模型；实验为比较不同方法，对全部 200 条都采集了第二模型的结果。部署费用只加总实际路径上的请求，时延按该路径串行调用的时延之和计算。最终输出是允许或拦截；实际向用户提问的次数和用户答复时间需要在交互实验中测量。[按方法计费](results/authorization_conditions/confirmation_arm_usage.csv)

| 本次实验费用（元） | 第一轮，含遮蔽与全部对照 | 第二轮，含全部对照 | 合计 |
|---|---:|---:|---:|
| Jev，服务商用量记录 | 0.591175 | 0.341419 | 0.932594 |
| DeepSeek，按 token 估算 | 0.840353 | 0.440596 | 1.280949 |
| 百炼，第一轮北京地域限时免费 | 0 | 未调用 | 0 |
| **本次合计** | **1.431528** | **0.782015** | **2.213543** |

每个阶段保留请求编号对应的本地样本标识、结构化答案、尝试次数、输入输出 token、缓存用量和客户端时间。价格按实验日人民币费率计算；工作簿和 CSV 中的有效位数高于上表。[第一轮账目](results/authorization_conditions/usage.csv) · [第二轮账目](results/authorization_conditions/confirmation_usage.csv) · [DeepSeek 价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/) · [百炼价格](https://help.aliyun.com/zh/model-studio/decision-model-preview/) · [Jev 用量查询](https://apikey.fan/key-usage)

已有原始数据、采样清单和模型凭据的工作副本可在 PowerShell 7 中运行以下命令。相同实验会复用已完成的调用。第二轮按类别抽取，适合检验这些动作的授权边界。线上合法与越权请求比例、授权长期有效性以及实际工具执行效果都需要另外测量。

```powershell
foreach ($provider in @('jev','aliyun','deepseek')) {
    ./scripts/study.ps1 -Study conditions -Provider $provider
}
foreach ($provider in @('jev','deepseek')) {
    ./scripts/study.ps1 -Study confirmation -Provider $provider
}
$py = uv python find
& $py authorization_results.py
& $py confirmation_results.py
```

[下载对比工作簿](results/authorization_conditions/授权条件复审实验对比.xlsx) · [第二轮分类型结果](results/authorization_conditions/confirmation_by_category.csv)

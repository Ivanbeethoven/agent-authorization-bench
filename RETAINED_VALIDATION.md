# 保留策略的新样本复测

每项任务取此前各批次至少保留过一次的策略的并集。纯 DeepSeek 在每条样本上给出独立的原生判断。各方法沿用上一轮决策提示、分歧转交条件和原生标签。对照使用相同输入。模型只接收决策时可见的字段，不接收当前题目的参考答案。专家示例沿用历史的 119 条参考库和跨家族检索。GPT 使用本地 Codex，每次独立判断一题。各阶段可复用于不同策略。

按历史试验计划、初评输入的编号和完整可见状态排除重复。Jev Boundary 优先在官方 test 切分按情景家族轮流抽取 50 条；家族可能已出现。AuthGateBench 的 700 条及 DelegationBench 的 149 条四分类共识题均已进入早期试验。为这两种任务各新编 25 对条件对照，每对固定候选行为，改变授权、事实或政策状态。同一批次中每对是一个实验单位。另对 DelegationBench 尚未使用且四分类无共识的公开记录，只使用发布者给出的二分类参考，单列成绩。未经专家标注的自建题不计作公开 benchmark 得分。

运行前固定一个按任务选择的方案：授权二分类使用 GPT‑6.2 Sol，四分类委托使用 DeepSeek 最小干预，策略三分类使用 Jev 直接判断。比较该方案与在三个任务上均入选的单一方法。对于通用方案，要求在每项任务上均达到纯 DeepSeek 的正确数和合法放行数，且误放行数不增加。先选择三项任务平均准确率最高者，平局依次比较平均误放行率较低、平均合法放行率较高、费用披露完整、总费用较低、典型用时较短。纯 DeepSeek 始终是可选方案。各任务另按相同条件分别列出本批最优。对已知金额的单项选择采用同样顺序。选出的方法属于本批结果，两项自建题只检验所列条件，公开争议题不参与选择。

记录原生准确率、误放行数及负例分母、合法放行、每条路径耗时中位数及第 95 百分位、人民币用量折算和可获得的账户差额。GPT 账户未披露逐次人民币金额，转交策略同时报告已知 API 部分与实际转交次数。时间按每条所需调用的实测时长相加；并发批次另计墙钟时间。比较表的各策略费用包含可共享的调用，实验新增支出按实际请求去重。

## 复现与查看

历史排除清单从早期运行的输入快照与初评请求读取。新增自建配对在 `finalist_cases.py`，受测方法和分组分配在 `retained_validation.py`。仓库中的逐条预测表包含样本编号、分组、参考标签、预测、耗时、已知 API 费用和所需 GPT 调用数。

准备好历史输入快照及公开数据后，在仓库根目录执行以下 PowerShell 命令可以复跑。已有成功请求会从本地运行记录读取。

```powershell
.\scripts\study.ps1 -Study retained_validation -Provider deepseek -Workers 10
.\scripts\study.ps1 -Study retained_validation -Provider jev -Workers 10
.\scripts\study.ps1 -Study retained_validation -Provider aliyun -Workers 5
uv run codex_judgments.py --retained --workers 8
uv run retained_results.py
uv run shared_report.py --from-tables
uv run --with matplotlib overview_report.py
```

只需查看或重建报告时，可直接打开已发布的 HTML，或执行上面最后两行，从已发布的表格与复测汇总生成网页。

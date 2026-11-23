# 哪些评测包含了全部样本？

在本项目使用的公开版本中，三项主要授权评测共有 1,126 条请求。逐一核对原始样本编号与预测编号后，三个评测都没有遗漏、额外样本或重复编号。纯 Jev、纯 DeepSeek 和两者联合的第一轮完整对照均为每条请求给出了最终决定。[逐条决定](results/full_permission/decisions.csv) · [分项成绩和评分分母](results/full_permission/scores.csv)

| 主要评测 | 已测 / 本轮目标 | 覆盖范围 |
| --- | ---: | --- |
| AuthGateBench | 700 / 700 | 全部 350 对请求，采用上游定义的不提供额外环境信号的条件。 |
| DelegationBench（DelegationCheck v1.0） | 156 / 156 | 五个子集全部纳入。156 条都有是否执行的二分类参考，149 条有四分类多数意见，因此四分类准确率以 149 为分母。 |
| Jev Boundary | 270 / 270 | 三种策略各 90 条。正式语料共 378 条，另外 108 条属于开发集。另行评测的 150 条历史请求单独计分。 |

在后来的实验中，DeepSeek 提取条件、程序检查参数，部分请求再交给 Jev 复核。该方法也对全部 1,126 条主评测请求给出了最终决定，其中 813 条进入模型复核，分别为 568、136、109 条。三个主要评测在项目早期已经使用过，论文将后来的对照列为重复评测。[方法与数据划分](NATIVE_EVALUATION.md) · [各方法的复核数](results/native_permissions/scores.csv)

## 本地 AgentDecision 分类器的 10,130 条记录

| 数据来源 | 纳入记录数 | 测试范围 |
| --- | ---: | --- |
| 上述三项主要授权评测 | 1,126 | 同一批公开授权请求。 |
| SteerBench-Work | 106 | 发布版的全部 106 个工作场景，每个场景评估一个决策点。 |
| XAAB | 180 | 使用版本中的全部 180 个静态场景，按提供的候选动作和策略描述评分。 |
| TS-Bench | 7,182 | 官方测试集的全部工具调用步骤。ASB-Traj 5,231 步、AgentDojo-Traj 1,220 步、AgentHarm-Traj 731 步。训练集 2,194 步、验证集 789 步另列。 |
| 项目自编办公请求 | 1,536 | 128 个场景族，每族 12 条，全部纳入。 |
| **合计** | **10,130** | 同一条记录进入复核时只计一次。 |

TS-Bench 按工具调用步骤计数，同一条轨迹可以贡献多个步骤。其中 166 条因本地模型上下文长度限制缩短了历史观察，保留了完整指令与候选动作，并计入成绩。XAAB 和 TS-Bench 的这一轮比较使用本地分类器与原始 AgentDecision 输出的重建结果。纯 Jev、纯 DeepSeek 的三项主评测成绩对应 1,126 条请求。[本地模型评分表](paper/source_data/local_reviewer/scores.csv) · [上下文处理明细](paper/source_data/local_reviewer/context.csv) · [比较方法](PERMISSION_HEADS.md)

## 早期每套 50 条的试验

Jev 与阿里云百炼在 11 个数据集上使用了相同的抽样，每套 50 条，共 550 条。AuthBench、AuthorizationBench、YoloFS-Ext、AgentDojo、AgentDAM、PrivacyLens 和 PrivacyLens-Live 在该轮均为抽样或从原任务提取的单步问题。例如，PrivacyLens-Live 从指定工具配置的 150 份邮件记录中选了 50 份。早期 AgentDojo 评估的是首个工具是否与任务相关，后来的 TS-Bench AgentDojo-Traj 评估的是每一步的工具调用风险。所有上述实验均根据记录判断候选动作，未实际执行这些动作。[逐数据集抽样数与题目类型](results/summary.csv) · [两个服务的对照结果](results/comparison.csv)

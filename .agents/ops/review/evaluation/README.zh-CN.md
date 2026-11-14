# 私有审查比较

[English](README.md) · [日本語](README.ja.md)

本工具实现[父 issue #1797 批准的比较](https://github.com/Motoki0705/tennis-lab/issues/1797#issuecomment-6786941605)。
收到单独执行指令后，已于 2026-11-11 串行完成最小方案的十四次调用。
额外调用、runtime/格式失败及用量缺失均为零。
[执行证据和父任务判定材料](../../../../knowledge/reports/review-model-comparison-20261111/README.md)
保留实测与父任务完成的问题判定。采用判断已记录于结果报告，正式审查实机验收仍待父任务完成。
常规 pytest 只运行离线契约测试，不启动比较。

## 固定资料与隔离

[manifest.json](cases/manifest.json) 固定每个案例的完整 head、base、merge-base、
diff 字节指纹、截止时间及输入指纹。缺少本地历史对象时直接失败，不自动 fetch。
[gold.json](cases/gold.json) 保存九个已确认的 major 问题、历史父任务确认与回归证据，
以及两个暂定负例的独立静态检查。历史测试结果归属于原确认者，并非本次重新执行。

| PR | 审查边界 | 正确性证据 |
|---|---|---|
| [#1699](https://github.com/Motoki0705/tennis-lab/pull/1699) | URL 凭据与原始日志发布 | 父任务确认、修复前后复现、ignore 规则检查 |
| [#1766](https://github.com/Motoki0705/tennis-lab/pull/1766) | 取消与自动继续竞争 | 父任务确认的取消回归测试 |
| [#1867](https://github.com/Motoki0705/tennis-lab/pull/1867) | 旧恢复覆盖后续问题、event/final 不一致 | 父任务确认的恢复与 runtime 测试 |
| [#1872](https://github.com/Motoki0705/tennis-lab/pull/1872) | 未确认 cleanup、首次 runtime 保存前中断 | 父任务确认的排他与启动边界测试 |
| [#1888](https://github.com/Motoki0705/tennis-lab/pull/1888) | 部分 reviewed 设置校验、handoff 丢失候选 | 父任务确认的部分结果与失败 final 测试 |
| [#1686](https://github.com/Motoki0705/tennis-lab/pull/1686) | memory 文档的人类指令归属 | 独立检查完整固定 diff 与引用指令 |
| [#1744](https://github.com/Motoki0705/tennis-lab/pull/1744) | 保存 player-view scene bundle | 独立检查 bundle、assets、CLI、坐标变换及测试断言 |

`inputs/` 是原始审查前 context 的精选投影。截止时间取原 runner 的启动时间。
保留父任务要求、目标子任务要求、PR 描述、变更路径、已保存的 CI 摘要与当时指令。
排除兄弟进度和 `parent-*` 运作评论；截止后新建或编辑的评论均不使用。
provenance 记录原 context 指纹、排除的评论 ID 与脱敏数量。
当时未保存的 PR/子任务讨论及 CI 原始日志保持不可用。

准备过程只复制固定输入与 diff，不向审查工具提供答案表、修复 commit、当前 checkout、
共享 memory 或后续评论。Git 工具只读固定 revision 及其祖先。
GitHub GET 只接受精确登记的 endpoint，其他请求返回 `UNREGISTERED_ENDPOINT`，不转发真实网络。
本地评论写入器不会发布，也不作为审查工具开放。复用现有有界检索和 journal。
正式 CLI 的权限、发布、排他及 gate 检查保持严格。

reviewer 在空的私有 git repo 中运行。专用 launcher 禁用 shell、图片、浏览器、apps/plugins、
hooks、memory、额外 agent、code mode、web 搜索、继承的 MCP 和 project 指令。
只接受共享 runner 参数及已登记的专用 MCP 配置。CLI 缺少必要控制时在启动前拒绝，
首次 CLI 版本也会固定。使用官方
[Codex 配置控制](https://learn.chatgpt.com/docs/config-file/config-reference)。
pilot 和后续事件记录只出现专用 MCP 调用。已观察到实际 model 和 effort；实际 service tier 仍不可用。
比较不提供 CPU 测试执行，工具环境与正式审查有差别。

## 准备和显式执行

检索服务器在构造时及每次调用前拒绝任何 `DECISION_LOG_RETRIEVAL` 或
`DECISION_LOG_CAPTURE` 设置，包括空值和直接 `serve` 入口。操作者必须先明确移除配置，
不会静默忽略配置，也不会把比较收据写入正式 parent journal。

使用 checkout 和正式 parent state 以外的专用 state。harness 输出、runtime、检索凭据和
本地写入使用私有权限；Codex 也可能保留其正常本地 transcript。不要提交原始 runtime/state。

```bash
export PYTHONPATH=.agents/ops
EVAL_STATE="${XDG_STATE_HOME:-$HOME/.local/state}/tennis-lab-agents/review-evaluation/minimum"
.venv/bin/python -m review.evaluation verify --repo .
.venv/bin/python -m review.evaluation init --state-dir "$EVAL_STATE"
.venv/bin/python -m review.evaluation prepare --repo . --state-dir "$EVAL_STATE" --case 1699
```

首次启动前，用 `record-selection --state-dir "$EVAL_STATE" --spec selection.json`
固定父任务的逐案例选择。spec 包含 `schema_version: 1`、非空父指令来源 `source`，
以及每个批准案例唯一的 `{case_id, model, reason}` 组成的 `choices`。
首次预约后不能更改。未记录选择时，在 CLI 检查和模型启动前拒绝执行。
父任务必须先固定条件 3 的逐案例选择，再消耗 pilot 调用。

数值契约以 [protocol.json](cases/protocol.json) 为准。先串行运行 pilot 案例的 Astra/Sol 各一次。
首次测量已可触发超量或失败停止。完成两次后，`status` 根据各模型 pilot 比率修正
[historical_usage.json](cases/historical_usage.json)，重估剩余组合及备用调用。
使用量缺失、pilot 无效或无法验证、预计超出预算时，停止并返回父任务。
token 预算是启动前估计，不是 API 的硬上限；单次运行仍可能超出。

仅在父任务另行指示执行后运行:

```bash
.venv/bin/python -m review.evaluation run --repo . --state-dir "$EVAL_STATE" \
  --case 1699 --model gpt-6-astra --parent-instruction-file parent-execution.txt --execute
.venv/bin/python -m review.evaluation status --state-dir "$EVAL_STATE"
# 随后执行 Sol pilot，检查重估，再逐次运行其余组合。
```

共享 runner 强制单次 timeout。预约持久化且串行；已收集的相同组合幂等返回。
预约已有文件时不重启，应先确认结束及 cleanup，再用 `collect --run-id ID`。
缺失或未确认 runtime 保留预约并阻止新调用，不可通过删除来恢复额度。
额外调用必须提供 `--additional` 和 `--reason-file`，失败启动同样消耗预约额度。
harness 不创建正式 review、decision capture 或 GitHub 评论。
初始化时固定代码、共享 runner 依赖、输入、答案表和 protocol 指纹。
更改比较须返回父任务；新建 state 不能重置已批准研究的调用和 token 预算。

## 指标和父任务裁定

`collect` 使用正式审查现有的 final、event、runtime/有效设置及 schema 校验。
runtime 失败、格式失败和不可用输出分别记录；证据未验证的输出不能作为有效审查计分。
wall time 从预约到 runtime 结束，不含人工收集延迟；agent 时间来自 runner 起止时间。

`adjudication-template --state-dir "$EVAL_STATE" --output adjudications.json`
输出问题、路径重叠候选和待定裁定。父任务比较触发条件、原因与影响，并以测试、复现或独立代码确认作依据。
路径相同不自动视为一致。父任务保留精确的输出和答案表指纹，填写 `adjudicator: parent`，
在 `matches` 中提供 `finding_id`、`gold_id`、`reason`、`evidence`；其余问题带理由和证据分类为
`false`、`unconfirmed` 或 `valid_new`。未填写模板不能作为完成裁定。
所有输出问题必须在 `matches` 或 `other_findings` 中恰好出现一次。遗漏、交叉分类、重复对应、
一个问题对应多个 gold 均拒绝；一个问题代表一个可独立修复的论点。多个问题对应同一 gold 时
只计一次。父任务未填写保持 pending；只有显式 `unconfirmed` 才计入无法确认数。

`report --state-dir "$EVAL_STATE" --parent-adjudications adjudications.json` 区分:

- 模型资格比较: 所有 Astra/Sol 物理调用，包括失败和额外确认；每个原始 counter 只算一次。
- 条件 1 `astra_fixed`: 新启动 Astra 的实测。
- 条件 2 `reuse_fixed`: 以相同 Astra 代码审查结果为质量参照，另算历史证据能否省略调用。
  不制造新模型结果或未测量的恢复延迟。
- 条件 3 `reuse_select`: 用事先选择投影配对样本，并计入所有实际额外调用。
  反事实投影不重复加到物理总量。联合发现需父任务裁定，不能直接相加各 run 的检测数。

检测和漏检按不同 gold ID 计数，区分 blocker/major。未裁定保持 pending 而非零。
runtime 失败和无效或缺失输出从质量计数及问题率分母排除，比较保持 incomplete。
失败 runtime 的可恢复 final 保留为调查候选 `candidate_only`，即使填写 match 也不算检测成功；
失败调用的实测资源仍全部计入。
runtime 失败计数包含 agent 启动前的失败；正常复用的 `not_invoked` 与失败区分。误报、问题无法确认、代码 `not_reviewed`、
证据等待、格式失败、runtime 失败及失败并集分别报告。
额外启动率为含多个独立调用的 episode 数除以已观察 episode 数。
资格比较的 episode 为案例/模型，选择投影则按案例计算。

保留 `agent-runtime` 的逐 turn 输入、缓存输入和输出。输入已包含缓存，输出已包含 reasoning，
不能再加一次。累计 counter 与缺失保持排除或未知，不补零。
无 AI 操作不能再次收取源 run 使用量；相同源 counter 去重，矛盾则拒绝。
不根据 token 推断费用或 quota 节约。矩阵完整且使用量已知后，才计算输入+输出或 agent 时间改善率。

## 公开投影与只读重算

`input_sha256` 始终标识实际测量的原始输入。无害化公开版通过 `input_publication` 记录不可变 Git
来源、原始和公开摘要、确定的替换规则和次数。`verify` 用本地 Git 历史重新执行转换并验证两个摘要。
私有原始 bundle 和每个 run 的原始输出摘要仍保留。包含历史公开投影的 `init` 和 `prepare` 被拒绝，
不会以改变后的刺激重新启动原实验。

代码更新后，完整的历史实验仅通过明确的只读路径重算：

```bash
.venv/bin/python -m review.evaluation report --state-dir "$EVAL_STATE" --repo . \
  --recorded-provenance knowledge/reports/review-model-comparison-20261111/provenance.json \
  --parent-adjudications knowledge/reports/review-model-comparison-20261111/adjudications.json \
  --output reanalysis.json
.venv/bin/python -m review.evaluation check-publication --repo . --base origin/main
```

此路径把原 ledger、代码、catalog、protocol 和私有输入绑定到公开 provenance，再验证保存的
final、event 和 runtime，并检查原 run、输出、counter 和时间不变。它不能启动、collect 或更新 ledger，
普通启动路径仍严格检查冻结代码。

公开检查覆盖 PR 全部变更文件，包括固定输入、裁定、README 和源码。JSON 使用原严格 Guard
及 worktree 检查；源码和文档检查具体 home/temp 绝对路径、native UUID 及已知 native 标识符，
允许系统可执行路径、schema 名和操作者模板。答案表中保留域 example.com 的凭证复现例及
repo-root ignore 规则按文档例验证，不豁免真实服务的凭证 URL。

## 历史复用与后续验收

[historical_reuse.json](cases/historical_reuse.json) 只读审计
[#1790](https://github.com/Motoki0705/tennis-lab/issues/1790) 的五次重复调用。
源与重复 snapshot 相同，现有 validator 能验证保存证据；但全部源结果为 `not_reviewed`，
因此当前恢复无法省掉这五次代码审查。额外使用量和时间是实际已发生工作，不是可节约量。
不升级 coverage，也不虚构证据等待。审计不包含当前权限、后续 request 和发布检查，
历史资格不能授权正式恢复。明确提供原 parent state 后可再生成:

```bash
.venv/bin/python -m review.evaluation audit-reuse \
  --parent-state-dir "$PARENT_STATE" --output reuse-audit.json
```

最小方案没有 gold blocker，且仅重复一次。暂定负例不证明不存在缺陷；缓存、顺序和工具差异仍有不确定性。
历史比较报告保留当时的策略和证据。当前正式审查策略、示例、用量查询及 fixed 回退以
[父级模型选择步骤](../../parent/README.md#正式レビューのモデル選択)为准。
此 harness 在部署设置为 fixed 或 select 时都显式选择比较模型，
保留已批准的 Astra/xhigh 参考设置和不可变证据。

另行授权后，父任务选正式 PR，核查 prepare/request、固定检索凭据、final/event、runtime 设置、
结果发布、父任务显式接收和 gate dry-run。确认未接收或缺少 CI 时 gate 阻止；适用时不重启代码审查完成证据。
在获授权且适用的 run 上确认复用、恢复和 Astra handoff，保留原问题及
[判断记录](../../decision_logs/README.zh-CN.md) 因果链接。
把实际 PR、CI、gate 证据及新的部署决定记录在父 issue。离线 fixture 不证明真实模型的质量或费用。

不要修改已固定或已消费的 request；
先核对活动 run、保留问题，再按显式指令创建替代 request 或 handoff。不加入自动重试和策略切换。

## 离线验证

用 `pytest -n 0` 顺序执行变更 unit 文件和两个判断链接 E2E node。
测试用小型本地 Git 历史及人工 runtime 证据，不要求历史 checkout、真实模型、GPU 或真实 GitHub。
覆盖未来资料隔离、未知 endpoint 拒绝、runner 参数、pilot 停止、额度、固定选择、缓存和源计数，
以及通过现有 decision-log 代码的发布投影与 README 生成。

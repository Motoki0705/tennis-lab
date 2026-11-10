# 父级判断记录

[English](README.md) · [日本語](README.ja.md)

本包提供结果、父级判断、行动和更正的共享版本1契约，保存私有快照并生成安全的公开单元。
捕获适配器观察现有parent、runner和review操作。Git发布和分层README生成属于 #1816。
记录本身不授予执行权限，也不判断成果是否通过验收。

## 契约与职责

字段、枚举和验证规则以代码为准，不在此重复维护。
现有[worker日志](../worker_logs/README.md)继续使用原来的三个文件；直接复用其schema工具、
`Guard`、拒绝重复键的JSON解码器、flock和持久化原子写入工具。

| 模块 | 使用契约 |
|---|---|
| [schema.py](schema.py) | `new_id()`、`validate_record()`、`json_bytes()`；严格的版本字段和缺失值表示 |
| [graph.py](graph.py) | `validate_graph()`、`references()`、`states()`；类型引用、DAG、行动历史和显式验证 |
| [journal.py](journal.py) | `Journal`；固定原始捕获和仅追加的准备/发布历史 |
| [projection.py](projection.py) | `render()`、`validate_public()`；公开单元 `{record, sanitization}` |
| [aggregation.py](aggregation.py) | `aggregate()`；保留所有原始记录引用的显示分组 |
| [capture.py](capture.py) | 私有outbox、行动快照、状态和仅捕获重放 |
| [parent_commands.py](parent_commands.py)、[parent_sources.py](parent_sources.py) | 父级判断入口、CLI结果、原始评论和显式验证 |
| [observations.py](observations.py)、[review_capture.py](review_capture.py) | 现有runner、dispatcher、worker日志、获取、review和merge事实 |

每条记录包含独立随机UUID4 hex ID、类型、归属、三个不同的时间和带类型的数据。
归属包含仓库、父issue、可空的子issue、公开活动键、现有worker公开委任键/轮次、
可空的公开review键。父级直接操作无需虚构子issue。
新ID和键必须使用 `new_id()`，不得从私有session/run ID或投递键派生。
对应关系只保存在捕获条目的私有context中。

`validate_record()` 严格检查嵌套键和类型，JSON输入使用共享解码器拒绝重复键。
`validate_graph()` 还拒绝缺失引用、类型错误、跨父issue引用、循环、行动历史冲突和不匹配的判断计划。
因果关系由引用表达，不从时间接近程度推断；支持一个结果分支到多个判断/行动，
以及多个结果合流到一个判断。

## 结果、判断与生命周期

结果区分机器观察、worker报告和父级验证。运行结束、成果质量、CI、最终报告是否可用、
发布情况是不同的观察。文件存在不代表父级已接收或接受。
显式父级验证是通过 `verifies` 引用受检结果的新结果。
`states()` 根据这些引用得出未判断和未记录验证的状态。

结果通过 `previous_result_id` 比较代码、需求、具体指令、执行条件和已验证范围。
没有基准时，所有差异必须是 `unconfirmed`。
证据包含固定revision或内容指纹；可编辑URL加指纹只能标识快照，不能保证重建全文。
私有或不可用证据保留原因，不暴露私有URL。

判断包含输入结果ID、简短理由、具体后续行动计划、复用/复查范围、未解决事项、恢复条件，
并区分设置固定、人类指令和父级选择的来源。
初始判断通过 `basis` 引用固定的请求/约定，不虚构前序结果。
计划也有独立随机公开ID，每个分支使用单独的计划和行动。

行动引用判断和计划。没有传入判断ID时，两者为null，并必须提供
`decision_missing_reason`，不能编造判断。
状态变化通过带 `previous_action_id` 的新行动快照追加，保持相同的归属、判断、计划和操作类型。
结果引用先前已请求/运行中的行动快照，之后的终止快照通过 `result_ids` 引用这些结果。
双方引用同一个快照会产生循环。`decided_not_executed` 行动不能产生结果。
行动历史在经过 `unknown` 后仍保留已确认的执行事实。
已经观察到执行的同一行动，不能退回已请求或启动前拒绝状态。

更正通过追加目标引用和理由表达，也可引用同类型、同归属的独立替代记录。
原始记录仍然存在，graph不会静默替换记录，也不会把更正当成新的观察。

## 缺失值、时间与用量

可用性只能是 `{status: observed, value: ...}` 或
`{status: unavailable, reason: ...}`。
未知时间、时长、用量和实际设置不得用0、所选模型或当前时间补齐。
真实观察到的0有效。发生/接收时间使用可用性对象，`recorded_at` 是真实记录时间。
所有已观察时间必须带时区。

所选model/effort保留指定来源，实际值独立观察。测量值保留证据。
用量保留单轮/累计范围和公开counter键；分组和本包都不累加累计计数、不估算费用，
也不把缓存token直接解释为收费量。

最终报告缺失时保留 `missing_final` 等原因。
未判断、未记录父级验证、已判断未执行、启动前拒绝、待发布由不同字段/状态表示，
不得从运行退出状态推断。

## 捕获入口与边界

结果的evidence注明 `capture_source`、artifact和观察快照的fingerprint。
私有证据仍明确标记为不可公开；fingerprint不等于全文档案。
已公开的父级评论、review和worker日志还保留可用的固定公开证据。
outbox只保存私有投递输入，不是发布单元。

| 入口 | 来源与边界 |
|---|---|
| `run-worker` | 父CLI请求、拒绝和响应；精确绑定runner的 `result.json`、选择/实际设置、取得的计数、结束原因和final可用性 |
| `start-worker` / `continue-worker` | CLI请求、拒绝和预留；dispatcher job绑定、持久化取消/runner返回及runner写入。预留不代表观察到启动 |
| dispatcher直接启动子任务 | 包括bootstrap子任务，使用相同job绑定和runner callback；不虚构父级判断 |
| `record-worker-run` / `record-worker-ended` | 显式登记已有runner证据 / 父级显式确认终止 |
| `record-result` | 已保存的验证与PR/证据字段；显式验证引用已捕获结果。组合命令 `review-request --prs` 使用相同方法 |
| `note` / `decide-worker-plan` | 原始评论引用 / 保存的计划回答结果。不会自动虚构通用判断，`decide` 可引用原始理由 |
| `delegate`、`stop` / `resume` / `stop-worker`、`handoff` / `release` | CLI结果和原始评论；dispatcher取消、runner停止/恢复是独立事实 |
| `review-prepare` / `review-request` / `review-state` | 固定请求及显式指示/状态结果；只有显式接收/验证才记录父级确认 |
| review `run` | 既有execution、output、result和发布状态，包括runtime失败时的有效输出、exception及先前review公开键。发布不改变原result；读取已消费run不是新AI评审 |
| review `recover` / `resume-publication` / `complete-evidence` / `resolve` | 既有保存证据恢复、仅发布重试和父级显式验证。保留前一结果，不算新AI启动 |
| review获取tool | 包括CI获取失败的既有receipt。同一review绑定、tool和参数的重试为 `retry_fetch`；只确认查询指示不变，不猜测网络等条件 |
| merge gate `merge` | 各PR判定、保存的merge尝试/结果及整体结果。区分dry-run合格和实际合并 |
| worker日志接收/发布、`worker-log-status` / `publish-worker-log` | 既有journal状态、final/发布失败和main保存确认。仅发布重试为 `retry_publication`，引用前一结果 |

支持的请求接受 `--decision-id` 和可选 `--plan-id`。只有一个匹配plan时自动选择；
拒绝歧义、操作类型不符或已请求的plan。缺少判断时两个ID为null，并明确保留缺失原因。
`--retry-of` 引用前一结果ID，`--changes-file` 接受契约的五个比较字段。
同一行动快照用 `previous_action_id`，独立尝试通过 `previous_result_id` 比较。
同一AI继续引用精确的前一runtime绑定。继续请求被解析为首次启动时，保留请求和独立的
`start_ai` 行动。即使调用方把默认模型展开为runner参数，也区分配置来源和显式选择。
保存的preflight还比较要求和发送指示快照，并检测启动revision变化。
相同commit不能证明工作代码未变；未观察的条件和验证范围保持未知。

未捕获：任意shell/HTTP/gh操作、这些CLI上下文之外的ParentSession直接调用、
没有父级绑定的runner，以及父级namespace确定前的参数解析、identity和仓库解析失败。
不重建过去对话或任意历史run。不更改review修复确认、恢复和模型选择策略；
捕获不会添加定期轮询、AI重启或Git发布。

## 显式记录父级判断

使用同一parent state和当前负责session：

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo OWNER/REPO --parent N decision-log-status
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo OWNER/REPO --parent N decide --spec tmp/decision.json
```

spec使用 `inputs`（一个或多个结果ID）、`evaluation`、`reason`，以及可选
`next_actions`、`reuse_scope`、`recheck_scope`、`unresolved`、`resume_conditions`、
`basis` 和 `source_kind`。plan指定schema的operation/summary/actor/target/instruction/
conditions/expected_result字段；CLI分配ID。初始判断须在 `basis` 提供固定公开请求/合意证据。
同一session重送相同spec和来源快照会返回相同判断/plan ID。
尚未请求的plan仍表示只作出判断、未实际执行。

可用一个 `reason_ref` 代替 `reason`：

| kind | 对原始来源的引用 |
|---|---|
| `note` | note命令返回的 `body_sha256` |
| `worker_plan` | `child` 和计划回答命令的 `decision_id` |
| `model_reason` | `child`；当前保存的模型选择理由 |
| `review_reason` | `run_id`；固定review请求的理由 |

判断保留引用/fingerprint，不再次复制公开理由正文。
父级actor键是私有session对应的随机公开键；交接不会改变旧判断的作者。
把返回的判断ID传给下一条支持的CLI（包括review/merge），后续结果通过行动与之连接。

## 捕获失败与恢复

向journal投递前，先在父级journal旁的私有 `captures/` outbox固定ID、时间、来源快照和投递输入。
flock串行化并发更新；调用方和runner重新读取当前outbox。
失败会向stderr报告并保留待投递记录。另一条私有诊断路径检测outbox写入失败，
以及尚未构造结果时的适配器失败。整个state filesystem损坏时只能通过stderr报告，
无法保证把诊断持久化到已损坏的介质。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo OWNER/REPO --parent N decision-log-status
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo OWNER/REPO --parent N retry-decision-capture
```

status列出来源、行动/结果ID、待投递/未结束receipt、缺失诊断、发布状态和无害化只读视图。
视图blocked不等于发布成功。重放只按依赖顺序投递固定记录，不重跑worker、review、获取或merge。
不可读receipt及丢失的来源快照保持未完成，不重置为空journal。
修复后可能需要重新登记已有的原始证据。观察失败不会更改原操作的退出值或状态。
显式 `decide` 或捕获恢复命令自身的记录/恢复失败时返回非零。

## 私有journal与发布边界

使用现有[parent state](../parent/README.md)创建
`Journal(parent_store.directory, repo, parent)`。
在 `decision-logs/` 下按仓库和父issue隔离，每个父issue一个journal。
目录为私有，文件/锁仅所有者可访问，拒绝符号链接。
条目数组只能增长，通过flock、文件/目录fsync和原子替换保护每次追加。
损坏JSON、重复/未知键、未知版本和归属变化会报错，不会重置为空状态。
这是本地文件系统契约，不是分布式归属锁。

- `capture(key, record, private=...)` 固定完整输入。私有context保存session、run、
  其他ID、checkout根目录和可空的checkout commit。
  可省略记录ID，由锁内分配UUID4；相同私有投递键重发返回相同ID且不写入。
  输入改变会被拒绝，调用方必须复用已固定的时间和归属。
- `prepare(record_id, key=..., at=...)` 固定安全公开单元，或追加 `blocked` 并保留原文。
  先准备依赖项。重试blocked需要 `retry=True` 和新投递键，不重新执行原工作。
  准备时还固定私有转换条件和完整性校验。读取journal时验证已保存的快照，
  不使用后来发现的私有ID重新render。ready的转换结果不能替换。
- `publication(record_id, key=..., status=..., at=..., evidence=..., error=...)`
  独立于捕获追加pending、blocked或published事实。
  published必须提供固定公开证据，是终止状态。
  Git/HTTP真实保存由调用方验证，本library不证明发布确已发生。
- `public_payloads()` 按依赖顺序返回验证通过的安全公开单元，包括无害化计数。
  发布方应使用这些单元并再次验证。`public_records()` 为graph/分组提供纯记录视图。
  `status()` 返回捕获、发布和缺失情况，不含私有对应关系。

导出以及pending/published发布记录会用当前已知的私有ID检查固定envelope。
后来发现的私有值可能阻止导出，但捕获内容仍保留。
调用方使用 `publication(..., status="blocked", error=...)` 记录发布失败，并保留原envelope。
共享worker日志guard的历史替换顺序保持不变。

共享worker日志guard先处理文字，再检查所有公开字段和无害化metadata。
已知私有ID和本地路径会被遮盖并记录替换计数。
固定引用URL、ID和模型metadata仅检查，不改写其含义。
残留凭据、未知UUID/session表示、危险URL和控制字符会阻止准备。
它不能完整识别未知秘密。Git路径、branch、生成README和PR正文仍需发布方的公开检查。

## 显示分组

`aggregate(records)` 只合并相邻的机器检查/获取重试，要求目标、归属、状态、行动、
条件键和模型设置一致，比较引用连续，且全部差异明确为未改变。
捕获方仅能对相同条件复用同一个随机条件键。
未确认/已改变条件、AI启动/继续、判断、行动快照和更正都会保留为边界。
同一获取行动内的尝试可以分组，不同的行动继续显示。
后续判断、验证、更正、行动结果或比较分支引用的中间观察也会保留，
即使引用出现在整个序列之后。只有下一条样本的简单比较链接可以被合并。
每组保留全部记录ID、数量、首次/最后记录时间和最后记录ID；使用方通过最后记录获取最终结果。
原始捕获和中间失败仍然保留，不补造时长或用量。

## 验证

```bash
.venv/bin/python -m pytest tests/unit/agents_ops/test_decision_logs.py \
  tests/integration/agents_ops/test_decision_log_journal.py \
  tests/unit/agents_ops/test_worker_logs.py -n 1 --durations=10
.venv/bin/python -m pytest tests/unit/agents_ops/test_decision_capture_observations.py \
  tests/integration/agents_ops/test_decision_capture_delivery.py \
  tests/e2e/agents_ops/test_decision_capture.py \
  tests/e2e/agents_ops/test_review_recovery.py -n 2 --durations=10
```

测试使用临时state覆盖分支/合流、显式缺失状态、拒绝案例、无害化、分组、
两个真实写入进程、fsync中断及损坏journal。捕获使用fake GitHub、临时git/state和offline runner，
并验证journal故障与仅捕获恢复；不需要GPU、真实代理或网络。
Git发布和合并后的实际运行验收属于后续任务。

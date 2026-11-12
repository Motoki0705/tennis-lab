# 父级判断记录

[English](README.md) · [日本語](README.ja.md)

本包提供结果、父级判断、行动和更正的共享版本1契约，保存私有快照并生成安全的公开单元。
捕获适配器观察现有parent、runner和review操作，并提供按父issue批量Git发布和确定性分层README。
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
| [navigation.py](navigation.py) | `generate()`、`validate_links()`；确定性分层生成与链接检查 |
| [presentation.py](presentation.py) | 将正文、旧summary中的结构值、变化和辅助观察转换为安全且有上限的显示字段 |
| [publication_policy.py](publication_policy.py) | 独立的仅追加例外、生成字节完全一致和PR身份检查 |
| [publication.py](publication.py)、[publication_state.py](publication_state.py) | 显式发布、私有批次和响应丢失后的核对 |
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
父级响应摘要只投影可公开字段，不整份复制assignment或任意嵌套状态。
私有来源ID在捕获前注册；完整响应与错误详情保留为私有证据。

| 入口 | 来源与边界 |
|---|---|
| `run-worker` | 父CLI请求、拒绝和响应；精确绑定runner的 `result.json`、选择/实际设置、取得的计数、结束原因和final可用性 |
| `start-worker` / `continue-worker` | CLI请求、拒绝和预留；dispatcher job绑定、持久化取消/runner返回及runner写入。预留不代表观察到启动 |
| dispatcher直接启动子任务 | 包括bootstrap子任务，使用相同job绑定和runner callback；不虚构父级判断 |
| `record-worker-run` / `record-worker-ended` | 显式登记已有runner证据 / 父级显式确认终止 |
| `record-result` | 已保存的验证与PR/证据字段；显式验证引用已捕获结果。组合命令 `review-request --prs` 使用相同方法 |
| `note` / `decide-worker-plan` | 原始评论引用 / 保存的计划回答结果。不会自动虚构通用判断，`decide` 可引用原始理由 |
| `improvement` / `improvement-publish` | 候选观察和明确的拆分判断引用原始评论，再连接起票请求/结果。仅捕获原父级仍负责期间的操作 |
| `agree` / `basis-record` / `create-child` / `workflow-report` | 现有依据、创建和验收结果。后继父级的agree通过来源评论记录观察到的人类批准，不将批准视为已执行 |
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
| `improvement` | `key` 和候选 `event_id`；原父级正本评论中的拆分判断 |

判断保留引用/fingerprint，不再次复制公开理由正文。
父级actor键是私有session对应的随机公开键；交接不会改变旧判断的作者。
把返回的判断ID传给下一条支持的CLI（包括review/merge），后续结果通过行动与之连接。

`improvement-decide` 在候选正本中记录人类判断，不扩大已结束原父级journal的发布权限。
后继父级通过agree导入批准时捕获观察结果。跨父级引用使用来源评论URL，不直接连接其他graph的记录ID。
操作与人类确认的信任边界见 [结束时改进](../parent/improvements.md)。

## 捕获失败与恢复

向journal投递前，先在父级journal旁的私有 `captures/` outbox固定ID、时间、来源快照和投递输入。
flock串行化并发更新；调用方和runner重新读取当前outbox。
失败会向stderr报告并保留待投递记录。另一条私有诊断路径检测outbox写入失败，
以及尚未构造结果时的适配器失败。整个state filesystem损坏时只能通过stderr报告，
无法保证把诊断持久化到已损坏的介质。
捕获初始化区分显式输入错误与journal/outbox故障。即使指定了判断ID，介质故障也不阻止
原操作，并留下持久的缺失诊断。仅重放投递不能把构造结果前已丢失的事实标记为恢复。

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

## Git发布与导航

父级记录判断或收到一组结果后，由当前负责人显式执行：

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo OWNER/REPO --parent N publish-decision-log --dry-run
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo OWNER/REPO --parent N publish-decision-log
# 核对并修复blocked原因后：
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo OWNER/REPO --parent N publish-decision-log --retry
```

没有定时器、轮询或每次捕获自动发布。入口使用已配置的parent state并检查负责session，
issue关闭后的历史发布也检查同一身份，不授予新的实现权限。
通过 `Journal.prepare()` 准备尚未转换的捕获，将 `Journal.public_payloads()` 中所有ready且
未发布的记录固定到该父issue的一个批次。blocked仍在状态中显示，独立安全的记录可以发布。
先核对未完成批次，再创建下一批；批次进行中新到达的捕获，在该批merge后的下一次显式调用中处理。

公开目录保留现有worker文件：

```text
.agents/logs/parents/
  README.md
  parent-<P>/
    README.md
    activity-<public-key>/
      README.md
      event-<public-record-id>.json
    child-<C>/
      README.md
      d-<existing-public-key>/
        README.md
        turn-<sequence>/
          README.md
        <sequence>-<initial|resume>/
          README.md
          # 现有instruction.md、final.md和meta.json不变。
```

每个event保存共享的 `{record, sanitization}` envelope原样内容。
README仅从该HEAD的公开记录和现有worker文件生成，不读取私有捕获/发布状态，也不调用LLM。
按因果引用排序，只有独立记录之间使用时间和ID确定顺序；分开显示活动，包括分支和合流。
中心表为 **時刻・対象 | 結果 | 親の判断と理由 | 次の行動・実行状態 | 根拠**。
结果、判断、行动快照和更正各自保留链接与反向引用。正文转义后作为文字显示，不能注入Markdown链接或HTML。
每页含上下层链接、活动入口、精确的记录成员和记录时间范围，明确显示未判断、已决定未执行和记录不足。
辅助时间、设置和用量放在折叠区，保留缺失和累计值，不相加。
稳定的 `turn-<sequence>/README.md` 不会根据序号猜测initial/resume。
worker文件稍后到达时，与原文件夹README双向链接；启动前拒绝后的initial重试也保留
先前公开的轮次地址。只链接指示和最终报告，不复制正文。

只为已有新记录的父issue生成历史轮次导航，链接现有指示和最终报告；没有对应记录时显示
**導入前・判断未記録**。只有旧worker日志的其他父issue仍链接原目录，不批量转换所有历史。
公开快照无法区分引入时间、待发布和捕获缺失时，明确保留为未确认。
失败轮次可以有README，但不能假装worker final已经发布。
公开快照无法统计私有待发布记录或证明自己已进入main，因此每页的 **公開待ち** 提示指向
`decision-log-status` 进行核对。merge后的发布事实本身不触发README重写。

### 概览布局与页面上限

[父issue列表](../../logs/parents/README.md)链接到父级概览。父、子、委任及轮次README显示
数量、缺失状态、最新观察结果/行动和最多六条近期判断，不重复全部记录。
判断同时显示输入结果、理由、复用/复查范围以及实际后续结果。
活动名称取自已记录的操作/命令和目标，记录期间单独显示。
不根据不透明活动键或相近时间推断名称、判断、验收或因果关系。
最终报告缺失数量单独列出来源明确标注 `not a final-report observation` 的非报告观察；
其他缺失原因仍然可见。未解决事项描述的是该判断记录时的状态。

五列表格和辅助信息位于活动页面。最多12条记录放在活动README；更多记录链接到同级
`activity-<public-key>-records-<sequence>/README.md`，并连接前/后页。
每条记录只有一个规范锚点，包括没有判断的常规机器观察，上层链接到相同锚点。
相同条件的观察可以显示既有的无损聚合数量，但不补造判断。

列表在概览内最多显示六项，完整目录每页40项。生成路径使用现有README模式：root下
`parent-index-*`、父级下 `activity-index-*`、子级下 `d-index-*`、委任下 `page-*`，
以及轮次文件夹的后缀。它们只是导航页，不代表虚构的父issue、活动、委任或轮次；event不移动。
按固定项目数分配页面，新增记录或worker文件后保留既有页面地址。
worker目录包含所有拥有worker日志的父issue，包括以后增加判断记录的父issue，因此新格式引入不会缩小该目录。
生成不读取既有README正文。

每个完整UTF-8页面必须小于192 KiB，超限则拒绝生成。每条记录的详细显示预算为10 KiB，
长正文明确标注省略，并链接完整不可变记录；不截断或改动原记录及固定envelope。
v1 summary中的dict/list literal通过有大小限制的literal解析显示为字段，不执行代码。
无法解析的结构提示查看原记录，不直接输出Python/JSON表示；解析得到的正文也全部转义。
解码后的结构、HTML entity还原的正文及格式化显示值，在HTML转义或省略前通过共用public `Guard`。
截断后的正文及带分隔符的字段名也重新检查；违规时以固定公开错误码拒绝生成，不遮蔽违规值后继续显示。

## 有限发布例外与恢复

这是独立于[worker三个文件例外](../worker_logs/README.md#専用prと限定merge例外)的条件。
专用branch为 `decision-log/parent-<P>/<random-batch-key>`；detached发布worktree只能位于
原仓库的 `.claude/worktrees/decision-log-<batch-key>-<revision>`。
每个revision只有一个commit，其唯一父级是固定main。
只有以下检查全部通过的专用PR，才可不要求Codex review、`Python` 或 `AGENT_AUTO_MERGE` 而创建/merge：

- 检查完整raw Git差异、旧/新mode和blob字节。新event只能使用规范父issue/活动/ID路径，
  并与固定journal envelope一致。旧event和worker文件不可更改；拒绝删除、rename、symlink、
  可执行文件、gitlink、额外文件和代码/规则变更。
- 检查schema、graph、metadata、公开正文、路径和链接。所需README必须全部存在，
  与该PR HEAD记录集合的再生成结果逐字节一致；拒绝任意追加正文、缺页或非生成README变更。
- 检查origin/repo、branch、固定marker、准确HEAD、main base ref、文件数以及GitHub所有分页的
  PR文件名/状态，并在merge前重新取得PR。

merge前fetch最新main，并从完整记录并集重新生成。main前进时创建新的已验证revision，
通过准确lease更新同一个PR branch，保留其他父/子issue的记录和链接。
连续观察到三次main前进时返回 `pending / main_advanced`，等待以后显式重试，不等待或轮询。
GitHub旧 `base.sha` 是快照，不能据此丢弃main更新。
按HEAD执行merge，检查实际merge tree的生成内容、main祖先关系和保存的envelope。
最后一次fetch与merge之间的外部更新也会在merge tree中检测；不一致时停在
`readme_refresh_required`，不宣称成功。显式 `--retry` 保留已merge PR的私有批次历史，
使用相同固定envelope建立 `-repair-<number>` branch。记录已在main时，修复PR只能改变
逐字节符合再生成结果的README，不会为了修复而添加/修改event。导航保存后再次运行不会创建更多PR。
批次merge后才到达的worker文件也使用同一显式修复，即使没有待发布的新捕获记录；不会补造捕获事实。
全部记录确认发布后的renderer更新也使用该路径。已保存的merge事实及批次全部记录的published事实
标识历史确认，仍重新检查PR/HEAD/merge身份、Git结构、祖先关系和当前envelope。
旧派生正文不必符合新renderer；main上的过时显示仍须显式 `--retry`，每个修复revision通过现行policy的完整检查。
未确认的旧批次不会因renderer变化而免除完整检查。

捕获journal旁的私有 `publication.json` 保存所有批次、revision、已尝试的变更和诊断，
采用flock、fsync和原子写入。`Journal.publication()` 只追加每条记录的pending/blocked/published事实。
发布命令本身不进入捕获适配器，因此不会生成下一轮发布记录。
commit后中断会复用完全核对的commit；push、PR创建和merge响应丢失时核对相同branch/marker/HEAD，
不盲目重发结果不明的POST/merge。open PR仍指向已知旧revision时，即使main再次前进，
也先恢复最新revision的本地commit和带lease的push，再要求最新HEAD一致。
所有revision均保留在私有历史中，包括从未应用于PR的revision。
已确认merge但逐记录journal追加中断时，先恢复该批再选择新批次。
不自动reset dirty worktree或损坏journal；只检查和修复受影响的发布。
确认后移除clean发布worktree，保留dirty或身份未确认的worktree，并返回 `retained_worktrees` 数量。

`--dry-run` 使用 `Journal.preview_publication()` 与本地 `origin/main`，保留ready envelope的固定版本，
用相同renderer/guard只读转换尚未准备的捕获，同时检查依赖。
返回 `prepared_records`、`preview_records`、`preview_blocked`（ID及错误码）、
`generated_readmes` 和 `preview_readmes`（路径到Markdown正文），首次发布也能检查内容。
预览被拒绝的记录计入blocked；`--retry --dry-run` 只预览blocked复查，不固定准备结果。
不进行不可逆准备、fetch、journal/ref更新、lock或worktree创建以及GitHub调用。
预览仅反映本地快照，新增捕获、guard上下文或main变化可能改变之后的实际发布。
只有当前所有捕获均确认发布，`saved_to_main` 才为true；批次merged仍可能剩余新捕获或blocked。
exit 0的pending不表示发布成功；有error或blocked返回exit 2。

准确workflow排除模式以 `ci.yml` 和 `pull-request-labeler.yml` 为准，跟踪允许规则以 `.gitignore` 为准。
范围是原三个文件、新event和各层README，不排除发布实现或任意raw日志。
`paths-ignore` 仅检查路径，不检查变更类型或内容。仅修改旧event、向README手写追加正文、
或混合新格式与worker三个文件的普通PR，也可能跳过CI和labeler，但不会因此取得有限发布例外。
专用发布命令检查完整内容与README再生成的逐字节一致性；常规 `merge_gate.py` 仍要求review及
该PR HEAD的 `Python` 成功，缺少check就拒绝merge。边界由这两项检查保证，跳过workflow本身不保证内容。
GitHub原生路径过滤限制仍然存在。
实际workflow省略及GitHub渲染在实现merge后的 #1817 中验证。

## 验证

```bash
.venv/bin/python -m pytest tests/unit/agents_ops/test_decision_logs.py \
  tests/integration/agents_ops/test_decision_log_journal.py \
  tests/unit/agents_ops/test_worker_logs.py -n 1 --durations=10
.venv/bin/python -m pytest tests/unit/agents_ops/test_decision_capture_observations.py \
  tests/integration/agents_ops/test_decision_capture_delivery.py \
  tests/e2e/agents_ops/test_decision_capture.py \
  tests/e2e/agents_ops/test_review_recovery.py -n 2 --durations=10
.venv/bin/python -m pytest tests/unit/agents_ops/test_decision_navigation.py \
  tests/e2e/agents_ops/test_decision_publication.py \
  tests/e2e/agents_ops/test_worker_log_publication.py -n 2 --durations=10
```

测试使用临时state覆盖分支/合流、显式缺失状态、拒绝案例、无害化、分组、
两个真实写入进程、fsync中断及损坏journal。捕获使用fake GitHub、临时git/state和offline runner，
并验证journal故障与仅捕获恢复；不需要GPU、真实代理或网络。
发布测试使用真实临时Git和fake GitHub验证批次、并发main更新、精确再生成、拒绝及响应丢失。
merge后的实际运行验收留给 #1817。

阅读验收测试位于 [test_decision_acceptance.py](../../../tests/unit/agents_ops/test_decision_acceptance.py) 和
[test_decision_log_acceptance.py](../../../tests/integration/agents_ops/test_decision_log_acceptance.py)。
覆盖参考 #1790 的三种异常、缺失状态、通常委任、父级直接验证、并行分支与交接，
以及[首次实际发布PR #1887](https://github.com/Motoki0705/tennis-lab/pull/1887)的副本和大量记录。
仅重试获取的分支明确标为假设场景：当时的CI事件实际进入了新review。
未记录的历史中间判断保持缺失；测量大小和验证保存在 #1817，
实现merge后由父级确认main发布及GitHub显示。

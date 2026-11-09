# 解释 HTML 契约、Gemini 编辑与本地服务

[English](README.md) · [日本語](README.ja.md)

本包为 [#1710](https://github.com/Motoki0705/tennis-lab/issues/1710) 提供共用文档/编辑契约和
持久化版本存储、visual import、受管 renderer、明确图像生成可用性的边界，
以及 Gemini 候选生成、独立语义检查和 loopback HTTP/job runtime。
浏览器 UI、图像 provider 实现和浏览器质量评估由外部注入。
使用现有的 Pydantic/Pillow/jsonschema/FastAPI/uvicorn 依赖和 Python 标准库。

## 契约正本与入口

[schema.py](schema.py) 是 wire 模型、HTTP 路径与响应类型、job 转移、公开错误和 JSON/Python
用例的唯一正本。下游直接 import；使用仓库 Python，设置 `PYTHONPATH=.agents/ops` 并运行
`python -m explain_html.schema` 可生成 OpenAPI 3.1。不要将 schema 复制到 adapter 或 web assets。
`contract_schema()` 在内存中返回同一契约；`examples()` 包含初始候选及故意未完成、不能 commit 的
验证结果。使用 `parse()` 严格解析 JSON，拒绝未知字段和类型转换。

| 模块 | 职责 |
|---|---|
| [schema.py](schema.py) | 文档、依据/assets/图、请求、候选、验证结果和 HTTP 契约 |
| [document.py](document.py) | ID、HTML 引用、选择范围检查和完整变更清单 |
| [store.py](store.py) | 初始 import、stage、commit、当前版/历史、恢复和导出 snapshot |
| [paths.py](paths.py) | 不跟随 symlink 的 POSIX 目录描述符操作 |
| [parsing.py](parsing.py) | 共用的歧义 HTML 语法和 CSS resource 记法拒绝边界 |
| [images.py](images.py) | 本地图像 import、能力报告与失败时停止的图像 adapter |
| [visual_definitions.py](visual_definitions.py) | 观测误差/比较图的严格声明式配置与数值模型 |
| [visuals.py](visuals.py) | SVG/HTML/CSS 安全检查、图 markup 和离线 renderer policy |
| [renderers/visuals-v2.js](renderers/visuals-v2.js) | 按声明 figure 绘图并隔离单图错误的当前 renderer |
| [renderers/visuals-v1.js](renderers/visuals-v1.js) | 为旧 revision 保留原始 bytes 的 renderer |
| [editor.py](editor.py) | 初次撰写、编辑、依据保持与独立语义检查 |
| [gemini.py](gemini.py) | 固定 model、禁用工具的 agy 执行、私有证据与取消 |
| [no_tools.py](no_tools.py) | 只允许 schema 检查后的完成操作的可信 PreToolUse hook |
| [jobs.py](jobs.py) | 持久化去重、文档队列、发布前检查与恢复 |
| [managed.py](managed.py) | 实 figure compile、图像不可用 adapter 和 staged visual 检查的 service 连接 |
| [service.py](service.py) | 共用契约的 HTTP route、token/Origin/Host 检查和公开文件 |
| [__main__.py](__main__.py) | 本地 start/serve/status/stop 和离线 export CLI |

## ID、目标与候选

生成方为可定位的逻辑块设置 `data-content-id`。同一逻辑内容保留 ID，新内容分配新 ID。
不固定标题、布局、配色、块标签或顺序。HTML 必须使用配对且显式的闭合标签，拒绝重复的
HTML/内容/成果物 ID。HTML 原生 `id` 与 `data-content-id` 属于不同命名空间。
图和依据通过 `data-figure-id`、`data-evidence-id` 引用，资源可以使用 `data-asset-id`。

共用 parser 拒绝有歧义或未完成的 comment、processing instruction，以及简单 HTML5 doctype 之外的声明。
comment 必须使用 `<!-- ... -->`，内部不允许连续 hyphen 或结尾 hyphen。`<!-->`、`<!--->`、`--!>`
结束、未闭合 comment 和无效 `<!...>` 声明均返回 `invalid_document`；document/visual 检查均不修复输入。
style 属性和 style 元素中的 CSS resource 记法也统一拒绝，包括 `url()`、
`image-set()`/`-webkit-image-set()` 字符串 URL、`image()` 和 `src()`。不支持的 CSS 被拒绝，
不会从引用归属和变更清单中静默省略。

每个请求用 `base_revision` 固定目标。内容目标必须存在。选区还包含 exact 引文、prefix/suffix，
以及整个块 DOM `textContent` 的 **UTF-16 code unit** start/end。按照
[HTML parser 规则](https://html.spec.whatwg.org/multipage/parsing.html#preprocessing-the-input-stream)，
先将输入 CRLF/CR 转为 LF，再展开实体引用，并移除 `pre`/`textarea`/`listing` 紧接着的一个 LF。
在此结果中计数，保留其他空白，非 BMP 字符占两个 code unit。存储和 markup 比较仍使用未改变的原始 HTML。
不是 text node 的局部 offset，也不是 `innerText`。位置、引文、上下文和 surrogate 边界必须全部匹配。
不会自动迁移引文或 rebase。

Candidate 包含完整的结果 Document、原始 EditRequest、内容变更、带原因的依据变更及摘要。
初始生成、局部编辑和整体重构使用同一契约；adapter 在 stage 之前将 patch 组装为完整候选。
初始 import 使用文档目标和 null base，后续更新必须引用当前版本。

`content_changes()` 列出块 markup 的新增/删除/修改，以及图定义、renderer 或所引用 asset 变更的块。
不依赖 ID 注释或 Figure，普通 `src`、`href`、`data` 及 `srcset`/`imagesrcset` 各候选也从 manifest
解析。每处引用必须属于某个 content ID，head 中的公共 resource 也一样；没有所属 ID 时返回
`broken_reference`。asset 变更通知所有包含该引用的 content ID。
仅移动未改变的块或更改外层布局时，允许空变更清单。`evidence_changes()` 列出依据新增/删除/修改。
stage 拒绝漏报、虚构或重复声明；依据变更的原因由独立语义验证判断，存储不会自行推测。

## 验证与发布

候选 digest 绑定请求、结果 HTML、全部 metadata、变更清单和每个 asset 的内容 digest。
`stage()` 保存私有候选并返回 StagedCandidate。可信验证方通过 `staged_candidate()` 和
`staged_asset_bytes()` 读取相同输入。asset 以 ID 为 key 提供 bytes；只有当前版本具有相同
asset ID 和 digest 时才可复用现有 bytes，缺失或不匹配会显式失败。每个版本的 Figure/Asset
记录同时固定 renderer 版本和 bytes。

`commit()` 只接受与候选匹配、structural/semantic/rendering 全部 passed 且无未解决 issue 的
ValidationResult。在文档级 filesystem lock 内再次检查结构、变更、base 和 bytes 完整性。
Candidate 和 HTTP 均不允许 writer 提交自己的通过判定。只有可信 controller 调用 commit；
判定记录不是 validator 执行或语义真实性的密码学证明。

controller 负责 HTML/SVG 清理、renderer policy、独立语义验证和渲染检查。结构检查解析 HTML 本地
引用和 fragment，允许外部引用链接，拒绝远程资源 URL 和带 URL 的 inline CSS。
srcset 接受 canonical managed URL 和可选的一致 `w` 或 `x` descriptor；无效、重复、混合 descriptor
或无法解析的候选都会使整个 source set 失败。
它不解释 asset 内部内容或执行 JavaScript。CSS/SVG/script 内容和图表专属定义由 visual validator
负责；仅通过结构检查不足以发布。

## Gemini 撰写与编辑

设置 `PYTHONPATH=.agents/ops`，使用 `explain_html.gemini` 的
`Gemini(private_root, agy=...)` 创建 `explain_html.editor` 的
`GeminiEditor(gemini, compile_document=compiler)`。必需的关键字参数类型为
`DocumentCompiler = Callable[[Document], Document]`。可信 compiler 展开图占位、插入
renderer/CSP 并返回最终 Document，或抛出 ContractError 拒绝；此 package 不隐式选择 visual 实现。
即使没有图，也必须用 `GeminiEditor(gemini, compile_document=lambda document: document)`
明确选择恒等变换，省略会报错。callback 收到 deep copy，返回值会重新验证并与可变状态分离。
生成 asset 的 bytes 由调用方 controller 私下保留并传给 stage；compiler 不得发布或 commit 文件。
受管 visual 可直接传入 `explain_html.visuals` 的 `compile_document=render_document`，
该公开函数的类型已经符合 DocumentCompiler。
`write_initial(request, context, *, writer_timeout=900, verifier_timeout=240, cancel=None)` 和
`edit(request, current, context, *, writer_timeout=900, verifier_timeout=240, cancel=None)` 返回 `EditorialResult`。
请求和当前版使用共享 EditRequest/Revision；初次请求使用 null base 和文档目标。
每个阶段的独立上限包含 effort/版本检查，writer 不消耗 verifier 的时间。
2026-11-09 实测初版 writer 为270–360秒、整体重构 writer 约633秒、普通 verifier 为60–110秒，
因此默认 writer 900秒、verifier 240秒。仅 Figure/slot 对应错误允许独立 writer 修复一次，上限600秒。
修复输入包括完整上下文、被拒候选、figure ID、预期 content ID、缺少slot/metadata或所属block错误。
修复后和compile后均重新执行正常检查；独立语义检查绑定最终digest，stage后仍须实际渲染检查。
修复再次失败即停止，语义、安全等其他失败不重试。prompt记录固定次数/时间，
`EditorialResult.repairs` 分别记录修复的model/effort/耗时/usage。
模型阶段合计最多29分钟（无修复19分钟），另加本地compile/渲染。
UI显示可取消的已用时间，不承诺完成期限，不应使用旧的共享600秒限制。
每次调用创建新workspace和会话，不resume或替换模型。

`WritingContext` 是可信 controller 输入：目的、读者、精确的目标依据集合、已批准 assets、
准备好的图、写作偏好和全部历史变更摘要。编辑时，当前与目标依据的每处差异都必须在
`authorized_evidence_changes` 明确声明。不要把 writer 的提案作为授权，也不要未经检查就接受
browser JSON 中的授权。独立 verifier 还会对照指令检查变更理由。两次调用都收到完整上下文、
当前版、请求/目标/指令和历史；verifier 另外收到候选及 digest。prompt/schema 通过私有文件和
stdin 传递，不因 argv 长度限制而截断。Candidate schema 直接从共享模型生成。
writer还收到`candidate_template`：按真实输出结构提供完整request和可信metadata，HTML/变更内容由writer填写。
这减少仅用ID代替依据对象或虚构request字段等native finish错误，不限定正文/布局，也不改变严格输出schema。
独立verifier只收到compile后的候选，不接收此写作辅助。

writer 返回完整 Candidate。机械检查绑定原始请求，执行共享结构/变更检查，严格核对目的、读者、
依据 metadata（包括数值、条件、来源）和已批准 assets。保留存续依据的正文引用和 HTTP(S) 来源链接。
未变更依据中已经出现在正文的数值，必须继续出现在该依据的范围内；此十进数字面检查不证明单位、
可见性、改写或语义。writer 检查后，必需的 compiler 可以展开 HTML 并添加生成 asset metadata。
editor 对照当前版重新计算 content changes，并再次检查结构和依据保持；保留原请求、依据变更理由
与摘要。只有最终候选会计算 digest 并传给独立 Gemini，检查这些方面及逻辑主张、已授权修改与缺少的输入。
可以生成诚实说明信息不足的候选。检查必须覆盖新旧全部依据 ID，绑定候选 digest，且无问题地 passed。
拒绝或无效输出会抛出 ContractError，不返回候选。模型语义检查提供额外证据，不保证真理或完整正确性。

writer 区分已提供的依据、用户明确给出的事实和一般定义/推导。主题请求或来源 URL 本身不能支持
产品规格、运动速度或实际精度。禁止添加未提供的数值输入、产品名及性能主张；基于已提供输入的计算
应在对应位置保留假设和假想/推导的性质。缺少信息时在正文说明，不猜测。

[editor.py](editor.py) 中 `SemanticVerdict` 必须包含 `unsupported_claims` 和 `technical_errors` 数组，
即使为空。每项含 `quote`、`reason` 和可选 `content_id`/`evidence_id`。verifier 检查整个文档，
包括依据标签外的正文、caption 和图表定义，列出无依据的具体主张、术语/坐标系混淆及无条件断言。
每项映射为 rule 为 `unsupported_claim` 或 `technical_error` 的 `ValidationIssue`，保存在
`ValidationResult.semantic.issues`。即使总判定为 passed，**任何一项都会导致失败**。
语义拒绝抛出 `SemanticValidationError`（仍是公开 code 为 `validation_failed` 的 `ContractError`）；
可信调用方可保存其绑定 digest 的 `.validation` 供私有诊断。不会返回候选，诊断正文不进入 HTTP 错误。
`schema.py`、保存的 ValidationResult 形状和既有 revision 不变。缺少两数组的旧模型判定无效，
应使用新 prompt/schema 重新检查，不填充空数组或复用旧批准。语义失败不自动修复/重试，重复拒绝也不放宽检查；
保持当前版，要求修正输入后显式重试。上述有次数限制的 figure 修复不变。此检查不保证完全正确，也不执行外部事实核查。

`EditorialResult` 包含 candidate、validation 和 writer/verifier 执行证据。validation 的
structural/semantic 为 passed，rendering 为 **not_run**，不能直接 commit。service 必须 stage
同一已编译候选和生成 asset bytes，对同一 bundle 执行独立 visual、renderer policy 和渲染检查，
再将真实渲染判定加入绑定 digest 的 ValidationResult。不得直接把未检查的 rendering 标为 passed。
会改变 HTML、assets 或 metadata 的展开与清理必须在语义检查前的 compile 中完成。
返回后修改候选需要重新验证，不能仅改 digest 来复用旧判定。compiler 拒绝时不返回 EditorialResult，
发布仍由 controller 负责。editor 不打开 store 或公开文件。

model/effort 明确固定为 `gemini-3.8-flash-high` / `high`，不同的 constructor 参数会被拒绝。
agy 版本来自既有 agents.conf 的 AGY_VERSION pin，不修改 issue 补充 runtime、pin 或权限。
每次记录选择值，要求只读 headless `/effort` 返回 `command.data.current=high`，并核对生成
stream 的 model 和全部 backend model override 日志。
[官方 effort 契约](https://antigravity.google/docs/cli/commands/effort/) 将 effort 定义为 model variant
选择；使用同一 High backend variant 和明确 effort 响应作为证据，不依赖模型自述或 token 数。
观测缺失或不符会返回 model_mismatch。

private_root 必须是配信/export 范围外的专用私有区域，目录 0700、文件 0600。
prompt、stdout、settings、schema、hook 记录、error 和 CLI 原始日志均保存在其中。
公开摘要仅限返回的数值用量、时间和已验证配置；异常 cause 和原始日志保持私有。
session strict 权限与 hook 只在按固定 schema 验证参数后，允许 agy 内部的 `finish` 工具。
该控制操作返回结构化数据；文件编辑、shell、外部调查和委派仍被拒绝。允许的完成参数必须同时
匹配最终 structured output 和已完成的 stream step。操作工具的任何尝试均使结果失败。
成功或失败都会核对 hook/settings/schema 完整性。这不是 OS sandbox，也不代表任意 global
plugin/hook 已被认定安全；请使用操作方管理的 CLI profile。adapter 不复制凭据或改写 user 设置，
agy 自身正常的 profile 行为仍然存在。

cancel 接收 `threading.Event`。service 在取消请求或停止时 set 同一个 Event；opsproc 在停止并回收
process group 后返回 cancelled。预先 set 的 Event 阻止启动。timeout、取消、model 不符、无效输出和
语义失败使用既有公开错误码。controller 仍需同步后续 commit 和取消。

## 本地服务与受信集成

设置 `PYTHONPATH=.agents/ops` 后构造 `JobController(store, editor_factory, validator, prepare=...)` 和
`create_app(controller, port=actual_bound_port, ui_files=...)`。CLI 始终绑定 **127.0.0.1**，默认使用
OS 分配端口，不提供 wildcard/LAN bind 选项。runtime root 由一个 controller 持有非阻塞 filesystem lock，
直到所有 worker 返回。不同文档可同时生成，同一文档串行执行。排队期间 base 过期的请求在生成前失败。
状态读取不会启动生成。

必须提供 `BundleValidator`。它接收 immutable staged Candidate、按 asset ID 索引的 **全部 staged bytes**
和 cancel Event，检查 HTML/SVG 安全、renderer 策略与实际渲染，返回 CheckResult。
controller 将结果绑定到 editor 的同一候选 digest，所有检查 passed 后才 commit。
没有内置渲染验证器或默认 passed 判定。仅做 visual 安全检查/figure compile 不能证明浏览器渲染。
应在独立语义检查前 compile figure；之后修改 HTML 会使该检查的 digest 失效。

`EditorFactory(prepared)` 为每个请求创建 editor。CLI 默认连接
[ManagedResources](managed.py)，在独立语义检查前执行真实 `visuals.render_document`，
展开图占位、插入 CSP 和版本固定的 managed script，没有恒等变换 fallback。
可选的 `--compiler module:factory` 是 managed compile **前** 的受信变换，不替换 visual/图像策略。
它接收请求独立的可变 `prepared.assets` map，并返回 DocumentCompiler；把生成的 SVG/renderer bytes
写入该 map，同一 bytes 进入 stage/验证，请求之间不共享变换或 bytes。

默认 `ManagedResources.prepare` 保留当前依据/assets 和完整变更摘要，把真实
`UnavailableImageGenerator.capability()` 的原因传给撰写和语义检查。
新的 AI 图像需求通过现有 Figure 类型表示：kind=image、generated_concept provenance、空 asset_ids
和空 figure 占位。compiler 通过 `generate_image` 将未满足图像意图传给不可用 adapter，在语义检查/commit 前
返回 `image_generation_unavailable`。移动现有图像或改说明时保留 asset ID。
自然语言解释由 Gemini 负责，service 不按特定语言的关键词匹配。自定义 `--prepare` 也必须保留此类型化约定。

CLI 先对全部 staged bytes 执行真实 `validate_visuals`，再调用指定的 BundleValidator。
后者必须执行实际渲染检查，不能把安全检查当作 rendering passed。

使用仓库 Python，把示例 validator 替换成实现 BundleValidator 的运营者受控 module：

```bash
export PYTHONPATH=.agents/ops
.venv/bin/python -m explain_html start --root outputs/explain-html --document triangulation --validator my_rendering:validate_bundle
.venv/bin/python -m explain_html status --root outputs/explain-html
.venv/bin/python -m explain_html stop --root outputs/explain-html
.venv/bin/python -m explain_html export --root outputs/explain-html --document triangulation
```

`serve` 用同样选项在前台运行。`--agy` 指定 adapter 的固定版本可执行文件，不更改 timeout/model/effort。
`--prepare module:callable` 和 `--ui module:callable` 加载受信本地代码，UI callable 返回固定 UI 文件名到
bytes 的闭合 map。浏览器请求不能选择 module。未指定 `--ui` 时 root 显示服务已启动的 placeholder。
start 等待 HTTP 启动后输出 localhost URL；status 通过 PID/boot/start identity 检查存活状态。
`--document` 验证现有文档，并将其 ID 放进启动 fragment。
stop 使用 Linux pidfd，仅向同一 identity 发 signal 并等待 shutdown/子进程回收。
export 可在服务停止后运行，输出 snapshot 目录；`--revision` 可指定旧的已发布版本。

启动 URL 在 **fragment 中将 session token 仅传给固定 UI**。UI 读取后移除 fragment，
将 token 仅保留在控制 UI 内存中，为 JSON API 添加契约的 `X-Explain-Token` header。
不得放入生成 HTML、iframe URL、query、cookie 或导出文件。每个进程创建新 token。
Host 只允许实际端口的 localhost/127.0.0.1；存在 Origin 时必须匹配；opaque origin `null` 不可调用
JSON API。CLI 可省略 Origin，但仍需 token。公开文档/export 无凭据，允许 opaque origin 资源请求。
CSP 强制不含 `allow-same-origin` 的 sandbox、禁止连接，仅允许版本固定的 renderer hash。
固定 UI 使用单独 CSP。

HTTP 路径、成功状态和公开错误直接从 `contract_schema()` 注册。提交立即返回 202，客户端按返回的
request ID 轮询 JobResponse；没有 SSE/WebSocket。同 request ID 和相同内容返回已保存的状态/结果，
包括终端失败；不同内容冲突。显式重试需新 ID 和正在显示的当前 base。
终端错误位于 JobResponse.error，接收拒绝使用 ErrorResponse。undo 创建新 revision，并将活动 job
标为 stale_revision。cancel 不回滚已 commit 的成功。

cancel、undo、进入验证和 commit 由 controller lock 同步。shutdown 设置传给 preparation/editor/validator 的
同一 Event，将活动请求标为 interrupted，等待 worker 返回后释放所有权。集成必须响应 cancel，并限制自身
渲染执行时间。重启将 queued/running/validating 恢复为 interrupted，不自动重发。
若 current pointer 已切换但 job 应答未保存，则严格匹配保存的 stage 和已发布 revision，恢复 succeeded。
journal I/O 不确定时，以明确错误停止编辑直到重启，不把已 commit 的版本误标 failed/interrupted。

undo 也在记录失败前核对发布状态：request ID、parent、restored_from 和完整目标文档必须匹配。
启动时也检查旧 runtime 记录了 error 的 restore journal。无法读取发布结果时停止工作，不判定成功。
export 的 rename/fsync 错误也仅在 manifest 与全部公开 bytes 核对通过时返回成功。

## 浏览器阅读与编辑

为 `start` / `serve` 添加 `--ui explain_html.web:files`，即可使用固定的阅读与编辑UI。
打开包含fragment的启动URL。UI立即移除fragment，仅在内存中保留token，并仅将其发送到需要
认证的JSON API。完整刷新页面后需再次打开启动URL。同一标签页的sessionStorage只保存待处理
request ID、操作类型、base和开始时间，不保存token或指令。重新连接按ID核对状态，绝不自动重发编辑或undo。
undo响应丢失时，历史尚未列出请求并不代表终态。保留request ID并继续核对发布历史，
保持当前版本可读，同时提供现有的重新检查状态操作。

通过目标菜单选择整页或内容block，也可点击标题/图表、按Enter/Space，或在同一block内选择文字。
选中文字与UTF-16位置会针对显示的revision核对。固定UI提供指令、cancel、耗时、公开错误、版本历史、
旧版预览与恢复；答案写入正文。明确重试使用新request ID和显示的当前版base。旧revision发生冲突时
保留正文，通过明确操作刷新当前版。

`GET /api/documents/{document_id}/revisions/{revision_id}/content/reader.html` 是已发布版本的临时阅读view。
opaque iframe不能向固定UI暴露DOM，srcdoc又继承固定UI的CSP，因此需要这一局部service扩展。
view加入最少的阅读辅助style和固定 `web/reader.js`，以其精确bytes的script hash连同版本固定的renderer hash授权执行。
临时response用HTTP CSP替换CSP meta，保留sandbox、connect-src none、resource限制和renderer hash。
共通content file_name枚举也声明reader.html，但JSON模型和export文件列表保持不变。存储HTML、validator、compiler、`content/index.html`与export bytes均不修改。

iframe使用 `sandbox="allow-scripts"` 和opaque origin，不授予same-origin、forms、popups或top-navigation，
也不传入token/cookie。生成script仍被阻止。固定UI仅接收来自确切frame window、origin `null`、当前
文档/revision及每个view nonce匹配的 `ready`、`selection` 与阅读 `state` message；内容ID/quote/位置及
state结构均与显示版本核对，无法通过message发送指令。阅读view使用共通StrictHTMLParser。目标索引在inert template中解析，不把节点插入固定UI的DOM。
此时app.js可能报告inline style属性被控制页CSP拒绝；保持该policy严格，正文style仅作用于iframe。
由于目标为opaque origin，父页面以target origin
`*` 发送连接、恢复状态/标记及导航数据。bridge核对父window及HTTP origin，不读取token也不访问service。

处理期间旧frame保持可读。成功发布的新版在第二个frame中初始化bridge后再切换。对于HTML、图表定义和
引用asset metadata相同的block，保留details与图表输入，包括block自身就是details的情况。
选择包含视口顶部的最内层可见block，按其内容ID和相对偏移恢复阅读位置，不focus/scroll
外部页面。变更以轻微内侧标记显示，只有明确按钮才导航。阅读锚点被删除时提示，并把原scroll高度限制在
新文档范围内；目标删除或改变后需重新选择才能发送。

预览旧版本时不可编辑，直到返回当前版或将旧版恢复成新revision。ZIP export下载service的精确snapshot文件，
不包含操作UI、bridge、认证信息或请求记录。解压后打开index.html即可offline阅读静态说明、图片和支持的图表。
AI图片只在初版生成；阅读期间可使用SVG、图表及已有图片，新图片请求以image_generation_unavailable说明未完成。

## 存储、恢复与导出

使用 Git 忽略的专用 runtime 目录，例如源码用例中的 `outputs/` 路径。整个 root 保持私有，
通过允许清单 API 提供公开文件。存储布局：

```text
<root>/private/documents/<document_id>/
  lock
  current.json
  stages/<stage_id>/{stage.json,public/...}
  revisions/<revision_id>/{record.json,public/...}
<root>/exports/<snapshot_id>/...
<root>/private/requests/<document_id>/<request_id>.json
<root>/private/{service.lock,service.log,session.json,gemini/...}
```

每个 revision 保存完整的 HTML、依据、图定义和 asset bytes。刷新文件与目录后，原子替换
`current.json`。替换之前失败保持原当前版；在替换边界发生 process/power loss 时，留下完整旧版或
完整新版。确认响应不确定时，重新打开并按 request ID 核对。失败的 pending 目录和不可达的已安装版
保持私有，不出现在历史中，`revision()` 也不返回它们。当前版损坏时不会静默回退旧版。
没有后台垃圾回收。

`current()`、`revision()`、`history()` 返回已发布版本，历史按新到旧排列。同一 staged candidate
成功后的重复 commit 是幂等操作，同 request ID 的不同内容冲突。`restore()` 将旧版完整文档复制为
**新 revision**，其 parent 为当前版，通过 `restored_from` 记录来源；重新验证所有 bytes，并复用同一
旧文档通过的 semantic/rendering 判定。旧 base ID 不会再次成为当前版。同一 restore 重试返回已创建版。

`snapshot()` 固定指定的已发布 revision，原子安装独立 bundle。公开文件仅包含 HTML、声明的 assets、
依据、图和公开 manifest。不会复制整个存储树、请求、验证结果、raw prompt、凭据或私有日志。
Document 中包括依据和来源的所有信息均用于公开，不要放入私有上下文。`asset_bytes()` 按 manifest ID
解析，`snapshot_file()` 按 snapshot 允许清单解析。拒绝绝对/父目录/编码路径、任何 symlink component
和 hard link。asset 路径统一为 `assets/<asset_id>.<ext>`。本实现面向包括 WSL 的本地 POSIX filesystem；
本子任务未验证 Windows Python、网络 filesystem 原子性或真实断电持久性。

## 图表与图像生成

定性场景/比喻使用生成图像，精确结构使用 SVG，数值比较或参数交互使用受管图表。
每个 Asset/Figure 都有 ID、caption、alt 和公开来源。受管 markup 将 caption、来源类别与出处一起显示；
图像 alt 和图表无障碍标签取自 Figure。上下文相关的图 caption 可以与复用 asset 的 caption 不同，
但图像/SVG 的来源必须一致。

`images.import_image(path, asset_id=..., caption=..., alt=..., provenance=...)` 返回带
`.asset` 和原始 `.content` 的 `ImportedAsset`。只接受可解码的单帧 PNG/JPEG/WebP，
上限为 16 MiB、16 megapixel。交互式 Codex imagegen 保存的图像也使用此入口，并标记为
`generated_concept`。`visuals.import_svg(content, ...)` 接受有 SVG namespace 和有限正 viewBox 的
受限几何/文字 SVG。拒绝 script、foreignObject、animation、style、entity 和资源引用，不静默删除。
SVG 作为 img 嵌入，不使用 object 或 active inline markup。

使用 `managed_renderer()` 的 Asset/bytes 和 Renderer version `2`，graph Figure.asset_ids
只包含此 renderer。`parse_graph()` 解析 Figure.definition；模型及其 `model_json_schema()` 是现有
wire 契约中图表配置的正本。拒绝未知字段、数值类型转换、非有限值及超出轴范围的数据。
仅支持 `observation_error` 和 `comparison`，没有公式/JS editor。前者明确为假想例，假定已经校正并
rectify 的 pinhole stereo 和正确对应关系，用 px 单位的 focal/disparity、m 单位的 baseline，
按 `Z = f B / (d + error)` 计算 m 单位的深度。带符号误差 slider、sample 输入、轴单位/范围与显示数据
必须一致。比较图保留最多八个命名系列，x 值严格递增且有明确单位/范围。
参见[具体 fixture](../../../tests/support/agents_ops/explain_html_visuals.py)。

在任意布局内放置空 `<figure data-figure-id="..."></figure>` slot，位于声明的 content block 内，
或让图本身作为该 block。需要 head/body 及显式结束标签。在**计算候选变更清单之前**调用
`render_document(document)`，填入图、CSP meta 和确切的受管 script。图中包含初始 SVG、源数据表、
假设、单位、输入范围与无障碍控件，禁用 JS 时仍能理解初始内容。同一 script bytes 既作为 Asset 保存，
也嵌入 HTML，由 CSP hash 授权。script 属于首个 graph 的 content block，在 DOM 解析完成后初始化
所有 graph，保留引用所有权。此方式避免 file:// 下外部 script 的 SRI/CORS 失败。snapshot 后不依赖 CDN、
fetch、module import 或 server。CSP 插入位置来自对 body 之前唯一 head 的实际解析，comment 中的 head
文字不能改变位置；policy 位于 head 中首个 script、style 和 resource 引用之前。不规范的 head/body 嵌套及
head 之前的非 ASCII 空白（包括实体形式的 NBSP）均被拒绝。

CSP、renderer script 和 `data-visual-style` 图表CSS属于compiler。
writer在编辑时也明确省略这些内容，无须复制代码或hash。同等CSP仅在解析后的directive名和token集合
完全符合所需policy时接受，再替换为正本。允许属性顺序、引号、http-equiv大小写、directive顺序和空白变化；
拒绝重复directive/policy、新增directive、来源变化及弱化policy。已有完全一致的managed script会被替换，
拒绝代码改动、额外属性和任意script。已有管理CSS也必须是正本。这是语义检查前明确声明的compile契约，
并非静默修补；stage后仍须与正本完全一致。即使writer设置`table{min-width:750px}`，管理CSS也会
在390px宽度把源数据表和图限制在figure内横向滚动。

公式使用Chromium原生presentation MathML，无需CDN或新增依赖。writer prompt与visual允许列表保持一致；
拒绝annotation-xml、mglyph、链接、外来元素和事件属性。拒绝TeX命令/分隔符，避免发布未渲染公式。
公式附近附有易懂说明，静态snapshot保留相同bytes。

TeX文本检查覆盖包含figure caption的正文，拒绝`$...$`、`$$...$$`、`\(...\)`、`\[...\]`及
反斜杠后接ASCII字母的所有命令。每个text run内连接inline元素、comment与展开后的字符引用，
防止拆分命令或分隔符绕过检查；block元素和literal区域结束run，不连接不同段落。
code/pre/kbd/samp/math的后代、属性值和head metadata不参与这项正文检查；源码记法与path放在literal元素中。
这些区域仍须通过MathML、HTML/CSS/script和属性的安全检查，不增加任何markup权限。
未配对的单个美元符号视为literal。若相邻的两个美元符号都引入数值价格，如`$5 and $10.50`，则视为货币。
金额允许符号、小数和分组分隔符，数字之后须为空白、标点或run末尾（下一个价格前的连字符也可分隔）。
仍拒绝`$5$`、`$x$2`和价格之后的数值公式；含糊的literal记法应使用literal元素。
编译和stage后的visual检查以`invalid_document`拒绝违规正文，不改写文本、不复用旧判定，也不新增TeX专用自动重试。

绘图属性（`data-managed-*`、`data-definition`、`data-plot`、`data-values`、`data-slider-value`、
`data-render-*`）仅保留给规范 graph figure；`data-graph-targets` 仅用于受管 script。
v2 读取 script 中的 figure/definition 清单，仅处理声明 figure slot 的直接子元素，并核对配置。
额外的匹配节点不会被当作 graph 输入。损坏的目标产生明确的图表错误，其他图继续初始化。

v1 bytes 保持不变，`managed_renderer("1")` 支持旧 Asset/Figure 的 snapshot/恢复，当前默认版本为 v2。
每个文档只能使用一个 renderer 版本，混合版本明确失败。未来实现使用新的 file/ID/version。
旧版 HTML/assets 从该 revision 读取，不注入当前代码。运行时检查只是额外防御，不能替代发布前检查和 iframe 隔离。

stage 后，从 store API 取得不变文档和**所有 staged asset bytes**，调用
`validate_visuals(document, all_asset_bytes_by_id)`。检查引用闭包、digest、MIME、asset 内部、受管
script 身份与规范图/data markup。拒绝 inline event handler、任意 script、嵌入浏览 context 和包含
URL 的 CSS。允许普通 CSS 及 media/supports/layer/container 规则；import、escape 和 CSS comment
不属于此保守子集。此确定性检查不生成语义/渲染通过判定。可信 controller 仍须在 commit 前运行独立的
语义/浏览器检查，并提供不带编辑凭据的 iframe sandbox。
也可通过 `content_security_policy(document)` 设置 HTTP header。

AI 生成图像**仅在创建初版时**通过交互式 Codex imagegen 生成，再从本地 import。
阅读期间 controller 使用 `UnavailableImageGenerator`；`capability()` 返回不可用，并说明仅限初版的
策略。阅读期间新增 AI 图像请求抛出共用错误 `image_generation_unavailable`，controller 必须将
**整个图像请求标为未完成**，中止发布并保留当前版。不能将局部文字/SVG 替代物作为图像生成成功发布。
现有图像 import、重新布局/caption 修改、SVG/graph 更新作为独立请求继续受支持。
这是[已确定的人类决策](https://github.com/Motoki0705/tennis-lab/issues/1710#issuecomment-6732169335)，
本范围内不再有待定的 service adapter 选择。

`generate_image(..., adapter=...)` 保留经过测试的 `ImageGenerator` 边界：能力检查与返回真实 bytes/公开
source 的生成。注入 adapter 用于成功/不可用/失败契约测试，阅读 service 不选择生成 provider。
provider/无效成果物失败抛出 `asset_generation_failed`，不公开 provider 内部细节，同样保持整个请求
未完成及当前版不变。不自动发现凭据或 provider。交互式 imagegen 的存在不等于本地 service 的生成
能力；调查证据保留在 PR 中。

## 下游职责

使用上述 runtime HTTP/发布边界，不要直接改公开文件。store 的 lock/base 比较也防止与受信外部 writer 冲突。
UI 负责 sandbox 展示、严格验证选区 message、变更导航和静态导出中的编辑不可用提示。
文档和所有 asset URL 必须绑定同一 revision，只有 succeeded job 返回新 revision 后才刷新。

## 验证

用仓库 `.venv/bin/python -m pytest` 执行 `tests/unit/agents_ops/explain_html` 和
`tests/integration/agents_ops/explain_html`。通过 `--collect-only` 统计数量，
`--durations=0 --durations-min=0` 测量时间。使用真实的小临时
文件、非对称块、多个 assets、声明式图表数据、异常注入和同步竞争 writer。
无需外部 API、GPU、浏览器或实时间 sleep。visual 新增 9 个 Python unit 和 3 个 Node case；Node 运行
`node --test tests/unit/agents_ops/explain_html/visuals.test.mjs`。
现有的恢复/snapshot 两个 integration case 使用真实 raster/SVG 和受管图表；另加一个 integration
验证 v2 发布后的 v1 snapshot/恢复，并固定原始 v1 bytes。浏览器验证共享一次 Chromium setup，两个场景为：
离线 slider/比较图/CSP 位置；拒绝无效语法/SVG/保留属性进入 export、保持当前版、隔离单图错误、
禁用 JS 时显示初始图表，以及旧 v1 的重放。

```sh
PLAYWRIGHT_MODULE=/path/to/playwright CHROMIUM_PATH=/path/to/chromium \
  node tests/e2e/agents_ops/explain_html_visuals_browser.cjs
```

harness 输出 setup/各场景时间和临时 screenshot 目录。
普通测试/CI 不调用真实图像生成或外部 API。
现有 JavaScript pytest 入口和 CI 通道也会运行这两个 visual suite。

editor 在 `test_editor.py` 添加 11 个 integration case，在
`tests/e2e/agents_ops/test_explain_html_editor.py` 添加 3 个公开入口 case。
使用可执行的假 agy、真实 store、完整长输入、允许/禁止的依据修改、拒绝/无效 verdict、真实 hook 拒绝
以及 pipe/Event 同步的进程回收。其中一例将真实的 `render_document` 和 stage 后的
`validate_visuals` 接入 editor 的语义检查与 store commit。editor 验证无需浏览器或固定的实时间 sleep。

runtime 在 `test_jobs.py` 添加 17 个真实 store integration case，在
`tests/e2e/agents_ops/explain_html/test_service.py` 添加 6 个 HTTP/CLI case。
使用假 editor/渲染判定、Event gate、OS 分配端口和私有临时 root，验证生成/commit 去重、请求独立的 compile bytes、过期 base、
cancel/undo/shutdown 后的延迟结果、活动 journal 恢复、pointer 切换后的 disk 错误、跨 session token、
Host/Origin/path 拒绝、公开 export 清单和 CLI 启停/重启/离线 export。
restore 覆盖 pointer 置换前后故障、无法读取发布结果和旧 error journal。
另有 3 个 store case 覆盖 export rename 故障和 readback 失败，在实际 directory fsync 调用处注入故障。

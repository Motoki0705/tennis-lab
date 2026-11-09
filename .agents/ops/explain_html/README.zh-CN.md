# 解释 HTML 文档契约

[English](README.md) · [日本語](README.ja.md)

本包为 [#1710](https://github.com/Motoki0705/tennis-lab/issues/1710) 提供共用文档/编辑契约和
持久化版本存储，以及 visual import、受管 renderer 和已验证图像 adapter 的边界。
不包含 Gemini 调用、HTTP 服务、浏览器 UI、图像 provider 实现或质量评估。
使用现有的 Pydantic/Pillow 依赖和 Python 标准库。

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
| [images.py](images.py) | 本地图像 import、能力报告与失败时停止的图像 adapter |
| [visual_definitions.py](visual_definitions.py) | 观测误差/比较图的严格声明式配置与数值模型 |
| [visuals.py](visuals.py) | SVG/HTML/CSS 安全检查、图 markup 和离线 renderer policy |
| [renderers/visuals-v1.js](renderers/visuals-v1.js) | 实现固定的受管 renderer version 1 |

## ID、目标与候选

生成方为可定位的逻辑块设置 `data-content-id`。同一逻辑内容保留 ID，新内容分配新 ID。
不固定标题、布局、配色、块标签或顺序。HTML 必须使用配对且显式的闭合标签，拒绝重复的
HTML/内容/成果物 ID。HTML 原生 `id` 与 `data-content-id` 属于不同命名空间。
图和依据通过 `data-figure-id`、`data-evidence-id` 引用，资源可以使用 `data-asset-id`。

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

使用 `managed_renderer()` 的 Asset/bytes 和 Renderer version `1`，graph Figure.asset_ids
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
fetch、module import 或 server。version 1 保持不变；将来的 renderer 使用新 file/ID/version。
旧版 HTML/assets 从该 revision 读取，不注入当前代码。

stage 后，从 store API 取得不变文档和**所有 staged asset bytes**，调用
`validate_visuals(document, all_asset_bytes_by_id)`。检查引用闭包、digest、MIME、asset 内部、受管
script 身份与规范图/data markup。拒绝 inline event handler、任意 script、嵌入浏览 context 和包含
URL 的 CSS。允许普通 CSS 及 media/supports/layer/container 规则；import、escape 和 CSS comment
不属于此保守子集。此确定性检查不生成语义/渲染通过判定。可信 controller 仍须在 commit 前运行独立的
语义/浏览器检查，并提供不带编辑凭据的 iframe sandbox。
也可通过 `content_security_policy(document)` 设置 HTTP header。

`generate_image(..., adapter=...)` 只返回真实的、已验证图像。默认
`UnavailableImageGenerator.capability()` 返回不可用及原因，不自动发现凭据或 provider。
可信且另行验证的 adapter 实现 `ImageGenerator`：能力检查与返回 bytes/公开 source 的生成。
不可用时抛出共用错误 `image_generation_unavailable`；provider/无效成果物失败抛出
`asset_generation_failed`，公开错误不包含 provider 内部细节。controller 在两种情况下都须将**整个图像
请求标为未完成**，中止发布并保留当前版。不能将局部文字/SVG 替代物作为图像生成成功发布。
现有图像 import、重新布局/caption 修改、SVG/graph 更新作为独立请求继续受支持。
交互式 imagegen 的存在不能证明本地 service adapter 可用；service 执行、认证及成果物回收必须另行
probe。能力证据与待定的人类决策记入 issue/PR。

## 下游职责

HTTP JSON 入口使用 `contract_schema()` 指定的 session-token header，不向隔离的文档/export 内容提供
编辑凭据。route 与 body ID 必须一致。内容文件与 asset 名必须匹配选定 revision 的允许清单，不得将
任意 HTTP path 拼接到私有目录。提供 HTML/assets 前先固定 revision ID，避免混合版本。
只返回 schema 定义的公开错误，不返回原始异常。

job controller 负责生成前持久化 request 去重、cancel/commit 同步、job 转移和 interrupted 恢复。
commit 时再次检查 cancel/revision，运行中保留旧视图，状态读取或重连不得重新生成。
store 的 lock/base 比较防止冲突发布，但不管理 subprocess 或 cancel 状态。
UI/renderer 负责 sandbox、选择事件、变更导航和静态导出中的编辑不可用提示。

## 验证

用仓库 `.venv/bin/python -m pytest` 执行 `tests/unit/agents_ops/explain_html` 和
`tests/integration/agents_ops/explain_html`。通过 `--collect-only` 统计数量，
`--durations=0 --durations-min=0` 测量时间。使用真实的小临时
文件、非对称块、多个 assets、声明式图表数据、异常注入和同步竞争 writer。
无需外部 API、GPU、浏览器或实时间 sleep。visual 新增 9 个 Python unit 和 3 个 Node case；Node 运行
`node --test tests/unit/agents_ops/explain_html/visuals.test.mjs`。
现有的恢复/snapshot 两个 integration case 使用真实 raster/SVG 和受管图表。
浏览器验证共享一次 Chromium setup，包含两个场景：离线 slider/比较图；拒绝恶意 SVG、保留当前版，
并在禁用 JS 时显示初始图表。

```sh
PLAYWRIGHT_MODULE=/path/to/playwright CHROMIUM_PATH=/path/to/chromium \
  node tests/e2e/agents_ops/explain_html_visuals_browser.cjs
```

harness 输出 setup/各场景时间和临时 screenshot 目录。
普通测试/CI 不调用真实图像生成或外部 API。
现有 JavaScript pytest 入口和 CI 通道也会运行这两个 visual suite。

# 説明HTMLの契約・Gemini編集・ローカルservice

[English](README.md) · [简体中文](README.zh-CN.md)

[#1710](https://github.com/Motoki0705/tennis-lab/issues/1710) の文書・編集の共通契約と、
永続的な版store、visual import・managed renderer・画像生成の可否を明示する境界、
Geminiによる候補生成と独立した意味検査、loopback HTTP/job runtimeを提供する。
ブラウザUI、画像provider実装、ブラウザ品質評価は別途注入する。
既存依存のPydantic・Pillow・jsonschema・FastAPI・uvicornとPython標準ライブラリを使う。

## 契約の正本と入口

[schema.py](schema.py) がwire型、HTTPパスと応答型、job遷移、公開エラー、JSON/Python使用例の
唯一の正本。下流から直接importする。repoのPythonで `PYTHONPATH=.agents/ops` を指定し、
`python -m explain_html.schema` でOpenAPI 3.1を生成できる。adapterやweb assetsへschemaを
複製しない。`contract_schema()` は同じ契約をメモリ内で返し、`examples()` は初版候補と、
commitできない未検証の判定例を含む。JSONには厳密な `parse()` を使い、未知fieldや型変換を拒否する。

| モジュール | 責務 |
|---|---|
| [schema.py](schema.py) | 文書、根拠・assets・図、要求、候補、検証結果、HTTP契約 |
| [document.py](document.py) | ID、HTML参照、選択対象、変更一覧の完全性 |
| [store.py](store.py) | 初版import、stage、commit、現在版・履歴、復元、export snapshot |
| [paths.py](paths.py) | symlinkを辿らないPOSIXのdirectory descriptor操作 |
| [parsing.py](parsing.py) | 曖昧なHTML構文とCSS resource記法の共通拒否 |
| [images.py](images.py) | ローカル画像import、能力の報告、失敗時に停止する画像adapter |
| [visual_definitions.py](visual_definitions.py) | 観測誤差例・比較グラフの厳密な宣言的設定と数値モデル |
| [visuals.py](visuals.py) | SVG/HTML/CSSの安全検査、図のmarkup、offline renderer policy |
| [renderers/visuals-v2.js](renderers/visuals-v2.js) | 宣言したfigureだけを描画し、図別に失敗を隔離する現行renderer |
| [renderers/visuals-v1.js](renderers/visuals-v1.js) | 旧revision用にbytesを固定して保持するrenderer |
| [editor.py](editor.py) | 初回執筆・編集、根拠保持、独立した意味検査 |
| [gemini.py](gemini.py) | model固定・tool禁止のagy実行、private証拠、中断 |
| [no_tools.py](no_tools.py) | schema検証済みの完了処理だけを許可するPreToolUse hook |
| [jobs.py](jobs.py) | 永続的な重複排除、文書別queue、公開前の再確認、復旧 |
| [managed.py](managed.py) | 実figure compile・画像利用不可adapter・staged visual検査のservice接続 |
| [service.py](service.py) | 共通契約からのHTTP route、token/Origin/Host検査、公開ファイル |
| [__main__.py](__main__.py) | ローカルstart/serve/status/stopとoffline export CLI |

## ID・対象・候補

生成側は編集対象となる論理ブロックへ `data-content-id` を付ける。同じ論理内容のIDを保持し、
新しい内容には新IDを付ける。見出し・配置・配色・ブロックのタグ・順番は固定しない。
HTMLは明示的で対応の取れた閉じタグを使う。HTML ID・内容ID・成果物IDの重複は失敗する。
HTML本来の `id` と `data-content-id` は別の名前空間。図・根拠は `data-figure-id`・
`data-evidence-id` で参照し、リソースには `data-asset-id` を使える。

共通parserは曖昧・未完了のcomment、processing instruction、単純なHTML5 doctype以外の宣言を拒否する。
commentは`<!-- ... -->`を使い、内部の連続hyphenや末尾hyphenを許可しない。`<!-->`・`<!--->`・
`--!>`閉じ・閉じないcomment・不正な`<!...>`宣言は`invalid_document`にし、document/visualの両検査で
補正しない。style属性・style要素内のCSS resource記法も共通に拒否する。`url()`のほか、
`image-set()`/`-webkit-image-set()`の文字列URL、`image()`・`src()`を含み、参照所有/変更申告から
静かに除外する代わりに未対応CSSとして拒否する。

すべての要求は `base_revision` へ対象を固定する。内容対象は存在必須。選択対象は、
完全一致の引用とprefix/suffix、ブロックのDOM `textContent` 全体における **UTF-16 code unit**
のstart/endを渡す。[HTML parser規則](https://html.spec.whatwg.org/multipage/parsing.html#preprocessing-the-input-stream)
に従い、入力CRLF/CR→LF、実体参照の展開、`pre`/`textarea`/`listing` 直後のLF1個の除去を経た
文字列で数える。その他の空白は保持し、非BMP文字は2 code unitとする。保存・markup比較用の
生HTMLは変更しない。text node内のoffsetや `innerText`
ではない。位置・引用・前後文脈のすべてとsurrogate境界を検査し、自動的な引用移動やrebaseはしない。

Candidateは更新後の完全なDocument、元のEditRequest、内容変更、理由付きの根拠変更、変更要約を持つ。
初版・部分更新・全体再構成は同じ契約。adapterが差分を完全な候補へ組み立ててからstageする。
初版は文書全体対象とnullのbaseを使い、以後の更新は現在版IDが必須。

`content_changes()` はブロックmarkupの追加・削除・変更と、図定義・renderer・参照assetが
変わるブロックを列挙する。ID注釈やFigureがない通常の `src`・`href`・`data` と、
`srcset`/`imagesrcset` の各候補もmanifestから解決する。すべての参照箇所がcontent IDの
配下にあることが必須で、head内の共通resourceも含め、所属がない参照は `broken_reference` にする。
asset変更は所属するすべてのcontent IDへ通知する。ブロックの移動や外側だけの配置変更では空の変更一覧も許す。
`evidence_changes()` は根拠の追加・削除・変更を列挙する。stageは未申告・架空・重複の変更を拒否する。
根拠変更の理由が妥当かは独立した意味検証が判定し、storeが推測しない。

## 検証と公開

候補digestは要求、更新後HTML、全metadataと変更一覧、全assetの内容digestを結び付ける。
`stage()` は非公開候補を記録してStagedCandidateを返す。検証側は `staged_candidate()` と
`staged_asset_bytes()` から同じ入力を得る。assetはIDをkeyとするbytesで渡す。
現在版に同ID・同digestがある場合だけ既存bytesを再利用でき、欠落や不一致は明示的な失敗にする。
Figure/Assetによりrendererの版とbytesも説明revisionへ固定する。

`commit()` はその候補に一致し、structural・semantic・renderingのすべてがpassedで未解決issueが
ないValidationResultだけを受け付ける。文書別filesystem lock内で構造・変更・base・bytesを再検査する。
CandidateやHTTPからwriter自身の合格判定を渡す入口はない。commitを呼ぶのは信頼するcontrollerで、
判定記録はvalidatorの実行や意味の真実性を暗号学的に証明するものではない。

controller側はHTML/SVGの無害化、renderer policy、独立した意味検証、表示検査を提供する。
構造検査はHTMLのローカル参照・fragmentを解決し、外部の出典リンクを許可するが、リモート
resource URL、URLを含むinline CSSは拒否する。srcsetはcanonicalなmanaged URLと、任意の
統一された `w` / `x` descriptorを受け付け、不正・重複・混在descriptorや未解決候補があれば
source set全体を拒否する。asset内部を解釈せず、JavaScriptを実行しない。
CSS/SVG/scriptの内容とグラフ固有設定はvisual validatorの責務。構造検査だけでは公開できない。

## Geminiの執筆と編集

`PYTHONPATH=.agents/ops` で `explain_html.gemini` の
`Gemini(private_root, agy=...)` を作り、`explain_html.editor` の
`GeminiEditor(gemini, compile_document=compiler)` へ渡す。必須のkeyword引数は
`DocumentCompiler = Callable[[Document], Document]` 型。信頼するcompilerが図slotの展開、
renderer/CSPの挿入を済ませた最終Documentを返すか、ContractErrorで拒否する。
このpackageがvisual実装を暗黙に選ぶことはない。図がなくても
`GeminiEditor(gemini, compile_document=lambda document: document)` と恒等変換を明示し、
省略はエラーにする。callbackにはdeep copyを渡し、戻り値も再検証して可変状態を分離する。
生成assetのbytesは呼出側controllerがprivateに保持してstageへ渡す。compilerから公開・commitしない。
managed visualには `explain_html.visuals` の `compile_document=render_document` を直接渡せる。
この公開関数の型はDocumentCompilerと一致する。
`write_initial(request, context, *, writer_timeout=900, verifier_timeout=240, cancel=None)` と
`edit(request, current, context, *, writer_timeout=900, verifier_timeout=240, cancel=None)` は `EditorialResult` を返す。
要求・現在版には共通のEditRequest/Revisionを使い、初回はnull base・文書全体対象にする。
各段階の上限はeffort/版照合を含み、writerはverifierの時間枠を消費しない。
2026-11-09の実機確認で初版writerは270〜360秒、全体再構成writerは約633秒、
通常verifierは60〜110秒だったため、既定をwriter 900秒・verifier 240秒とした。
Figureとslotの対応漏れだけは独立したwriter修復を最大1回・600秒で行う。
完全な文脈と不採用候補、figure ID、期待するcontent ID、slot/metadataの欠落・所有block違いを渡す。
修復後とcompile後も通常の全検査を行い、独立意味検査はcompile後のdigestへ結び付け、
stage後の描画検査も必須とする。再失敗は停止し、意味・安全性など他の失敗はretryしない。
promptに固定の回数・時間上限を残し、`EditorialResult.repairs` に修復のmodel/effort/時間/usageを別記する。
model段階の最大合計は修復あり29分・なし19分で、別途ローカルcompile/描画が必要。
UIは中止可能な経過時間表示で完了期限を約束せず、旧600秒の共有期限を設けない。
呼出ごとにworkspaceと会話を作り直し、resume・別modelへの代替はしない。

`WritingContext` は信頼するcontrollerの入力で、目的・読み手・更新後の正確な根拠集合、
許可済みassets・準備済み図・文章の好み・過去の変更要約すべてを渡す。
編集では現在版と根拠集合が異なる箇所を `authorized_evidence_changes` にも明示する。
writerの案を承認の根拠にしたり、browser JSONを未検査で承認として扱ったりしない。
独立verifierは変更理由と指示の整合性も調べる。両呼出へ完全な文脈・現在版・対象と指示・履歴を渡し、
verifierには候補とdigestも渡す。prompt/schemaはprivate fileとstdinで渡し、argv上限による省略をしない。
Candidate schemaは共通モデルから直接生成する。
writerには`candidate_template`も渡す。正確なrequestと信頼するmetadataを出力と同じ構造に入れ、
HTML・変更一覧はwriterが埋める。根拠をID文字列へ省略したりrequest fieldを捏造するnative finishの
誤りを減らすためで、本文・見た目・厳密な出力schemaは変えない。独立verifierへはこの補助を渡さず、
compile済み候補を渡す。

writerは完全なCandidateを返す。機械検査は元要求の一致、共通の構造・変更検査、目的・読み手・
根拠metadata（数値・条件・出典を含む）・許可済みassetsの厳密な一致を確認する。
存続する根拠の本文参照とHTTP(S)出典リンクを保持する。変更しない根拠の数値が従来の本文に
現れていれば、その根拠の範囲に残っていることも調べる。この十進数の字句検査だけでは単位・
表示上の可視性・言い換え・意味は証明できない。このwriter検査の後で必須compilerがHTMLを展開し、
生成assetのmetadataを追加できる。editorは現在版との差分からcontent changesを計算し直し、
構造と根拠保持を再検査する。元要求・根拠変更理由・要約は保持する。
この最終候補だけをdigest化し、別のGemini呼出へ渡し、これらや論理的主張・
許可された変更・不足する情報を意味検査する。情報不足を正直に説明する候補も生成できる。
検査は旧/新根拠の全IDを対象にし、候補digestが一致し、未解決issueなしでpassedを返す必要がある。
拒否・不正出力は候補を返さずContractErrorにする。意味検査は追加の証拠であり、真実性や完全な正しさの保証ではない。

writerは提供根拠・ユーザーが明示した事実・一般的な定義/導出を区別する。題材の説明依頼や出典URLだけでは
製品仕様・スポーツの速度・実環境の精度の根拠にはならない。未提供の数値入力・製品名・性能を加えず、
提供入力からの計算は前提と試算/導出の区別を該当箇所に保持する。不足情報は推測せず本文で説明する。

[editor.py](editor.py) の `SemanticVerdict` は空でも `unsupported_claims` と `technical_errors` 配列が必須。
各findingは `quote`・`reason` と任意の `content_id`/`evidence_id` を持つ。verifierは根拠タグ外の本文・
caption・グラフ定義を含む文書全体で、根拠外の具体的主張、用語・座標系の取り違え、正当化されない断定を調べる。
列挙した各項目を `unsupported_claim` / `technical_error` というruleの `ValidationIssue` として
`ValidationResult.semantic.issues` に残す。全体判定がpassedでも **1件あれば不合格**。
意味不合格は `SemanticValidationError`（公開codeは従来どおり `validation_failed` の `ContractError`）で、
信頼する呼出元は `.validation` のdigestに結び付いた結果をprivate診断用に保存できる。候補は返さず、診断本文は
HTTPエラーに含めない。`schema.py`、保存するValidationResultの形、既存revisionは変えない。
2配列がない旧model判定は無効とし、空配列を補ったり旧承認を流用せず新prompt/schemaで再検査する。
意味不合格の自動修復・再試行や繰り返し失敗後の検査緩和は行わず、現行版を保持して修正入力による明示再試行を求める。
前述の上限付きfigure修復は変更しない。この検査は完全な正しさを保証せず、外部調査による事実確認も行わない。

`EditorialResult` はcandidate・validation・writer/verifierの実行証拠を持つ。
validationはstructural/semanticがpassed、renderingは **not_run** のため、そのままcommitできない。
serviceはcompile済みの同じ候補と生成asset bytesをstageし、同じbundleへ独立したvisual・
renderer policy・描画検査を実施して、実際の描画判定をdigestに結び付いたValidationResultへ加える。
描画結果を確認せずpassedにしない。HTML・assets・metadataを変える展開や無害化は意味検査前の
compileで行う。返却後の候補変更には再検証が必要で、digestだけを書き換えて旧判定を流用しない。
compile拒否ではEditorialResultを返さず、公開はcontrollerに任せる。editor自身はstoreや公開ファイルを開かない。

model/effortは `gemini-3.8-flash-high` / `high` を明示し、異なるconstructor指定は拒否する。
agyの版は既存agents.confのAGY_VERSION pinを使い、issue補足用runtime・pin・権限は変更しない。
毎回選択値を記録し、読み取り専用のheadless `/effort` が `command.data.current=high` を返すことと、
生成streamのmodel・バックエンドへの全model override logが指定どおりであることを照合する。
[公式effort契約](https://antigravity.google/docs/cli/commands/effort/) はeffortをmodel variantの選択と定義する。
同じHigh variantのbackend logと明示的なeffort応答を証拠とし、モデルの自己申告やtoken数で補わない。
観測欠落・不一致はmodel_mismatchとする。

private_rootは配信・export対象外の専用領域にする。directoryは0700、fileは0600。
prompt・stdout・settings・schema・hook記録・error・CLI生ログをそこへ残す。
公開できる集計は返された数値の使用量・時間と検証済み設定に限定し、例外のcauseや生ログはprivate扱いにする。
sessionのstrict権限とhookで、agyの内部 `finish` toolだけを引数の固定schema検証後に許可する。
これは構造化データを返す完了処理であり、ファイル編集・シェル・外部調査・委任は引き続き拒否する。
許可した引数は最終structured outputと完了済みstream stepにも一致する必要がある。
操作toolの試行がある結果は失敗にする。hook/settings/schemaの完全性は成功・失敗とも照合する。OS sandboxではなく、
任意のglobal plugin/hookを安全と認定するものでもない。運用者が管理するCLI profileを使う。
adapter自身は認証をコピーせずuser設定を書き換えないが、agy本来のprofile動作は残る。

cancelには `threading.Event` を渡す。serviceは取消や停止時に同じEventをsetし、opsprocが
process groupの停止・回収を終えてからcancelledを返す。起動前にset済みなら起動しない。
timeout・cancel・model不一致・不正出力・意味検査失敗は共通の公開エラーを使う。
後段commitとcancelの同期は引き続きcontrollerが担当する。

## ローカルserviceと信頼する統合部品

`PYTHONPATH=.agents/ops` で `JobController(store, editor_factory, validator, prepare=...)` と
`create_app(controller, port=actual_bound_port, ui_files=...)` を構築する。
CLIのbindは常に **127.0.0.1**、portの既定はOS割当で、wildcard/LANへのbind指定はない。
runtime rootは非待機のfilesystem lockで単一controllerが所有し、全workerの終了まで保持する。
異なる文書は生成を並行実行し、同じ文書の要求は直列実行する。queue中にbaseが古くなった要求は
生成前に失敗する。状態取得は処理を起動しない。

`BundleValidator` は必須。immutableなstaged Candidate、asset IDで引ける **全staged bytes**、
cancel Eventを受け取り、HTML/SVGの無害化、renderer方針、実際の描画を検査してCheckResultを返す。
controllerがeditorの候補digestへ結び付け、全検査がpassedの場合だけcommitする。
組込みの描画validatorや既定のpassed判定はない。visualの安全検査・figure compileだけでは
browser描画の証拠にならない。figureは独立した意味検査より前にcompileする。
意味検査後のHTML変更はそのdigestを無効にする。

`EditorFactory(prepared)` は要求ごとにeditorを構築する。CLIは既定で
[ManagedResources](managed.py) を接続し、独立した意味検査より前に実 `visuals.render_document` で
図slotを展開し、CSPと版に固定したmanaged scriptを入れる。恒等変換へのfallbackはない。
任意の `--compiler module:factory` はmanaged compile **前** の信頼する追加変換で、visual/画像方針を
置き換えない。要求単位の可変 `prepared.assets` mapを受けてDocumentCompilerを返し、生成した
SVG/renderer bytesをそのmapへ入れる。同じbytesをstage/検証へ渡し、要求間の混在を防ぐ。

既定の `ManagedResources.prepare` は現在の根拠・assets・全変更要約を保持し、実
`UnavailableImageGenerator.capability()` の理由を執筆/意味検査へ渡す。
新規AI画像の要求は既存Figure型で、kind=image、generated_conceptのprovenance、空asset_ids、
空figure slotとして示す。compilerはこの未充足の画像意図を `generate_image` 経由で利用不可adapterへ
渡し、意味検査/commit前に `image_generation_unavailable` とする。既存画像の移動や説明変更は
asset IDを保持する。自然文の解釈はGeminiが担い、serviceは言語別の単語一致では判定しない。
独自の `--prepare` もこの型付き未充足要求の規約を保持する。

CLIは全staged bytesへ実 `validate_visuals` を実行した後、指定したBundleValidatorを呼ぶ。
後者には実描画の検査が必要で、安全検査だけをrendering passedにはしない。

repoのPythonで実行する。例のvalidator名はBundleValidatorを実装する運用者管理のmoduleへ置き換える。

```bash
export PYTHONPATH=.agents/ops
.venv/bin/python -m explain_html start --root outputs/explain-html --document triangulation --validator my_rendering:validate_bundle
.venv/bin/python -m explain_html status --root outputs/explain-html
.venv/bin/python -m explain_html stop --root outputs/explain-html
.venv/bin/python -m explain_html export --root outputs/explain-html --document triangulation
```

`serve` はstartと同じoptionでforeground実行する。`--agy` はadapterのpinされた実行ファイルを
指定し、timeout/model/effortの規則は変えない。`--prepare module:callable` と
`--ui module:callable` は信頼するローカルコードを読み込む。UI callableは固定UIのfilenameから
bytesへの閉じたmapを返す。ブラウザ要求からmoduleを選べない。`--ui` なしではrootに起動済みの
placeholderを表示する。`--document` は既存文書を検査し、起動fragmentへそのIDを入れる。
startはHTTP起動後にlocalhost URLを出力し、statusはPID/boot/start identityで
生存確認する。stopはLinux pidfdでそのidentityだけへsignalを送り、shutdownと子process回収を待つ。
exportはservice停止中にも使え、完成したsnapshot directoryを出力する。`--revision` で過去の公開版を選べる。

起動URLは **session tokenをfragmentで固定UIだけへ** 渡す。UIはfragmentを読み取ってURLから除き、
tokenを操作UIのメモリだけで保持し、JSON APIへ契約の `X-Explain-Token` headerを付ける。
生成HTML・iframe URL・query・cookie・exportへ渡さない。processごとにtokenを作り直す。
Hostは実port付きlocalhost/127.0.0.1に限定し、Originがあれば一致を要求する。
opaque originの `null` はJSON APIを呼べない。CLIはOriginを省略できるがtokenは必要。
公開本文/exportにはcredential不要で、opaque originのresource取得を許可する。
CSPは `allow-same-origin` のないsandboxで接続を禁止し、版に固定したrenderer hashだけを許可する。
固定UIは別のCSPを使う。

HTTPパス・成功status・公開エラーは `contract_schema()` から直接登録する。送信は即座に202を返し、
返されたrequest IDでJobResponseをpollする方式。SSE/WebSocketはない。
同じrequest ID・同じ内容なら失敗を含め保存済み状態/結果を返し、異内容は競合する。
明示的なretryは新IDと表示中の現在baseを使う。終端エラーはJobResponse.error、受付拒否はErrorResponseに入る。
undoは新revisionを返して進行中jobをstale_revisionにし、cancelはcommit済みの成功を巻き戻さない。

cancel・undo・検証開始・commitはcontroller lockで同期する。shutdownは準備/editor/validatorへ渡す
同じEventを設定し、進行中をinterruptedにしてworker終了後に所有権を解放する。
統合部品はcancelへ応答し、描画処理自身に上限を設ける。再起動はqueued/running/validatingを
自動再送せずinterruptedへ回収する。現在版切替後・job応答保存前の中断は、保存済みstageと公開版を
厳密に照合してsucceededへ回復する。journal I/Oが不確実になったら再起動まで明示エラーで編集を止め、
既にcommitした版をfailed/interruptedと誤表示しない。

undoも失敗を確定する前に公開状態を照合し、request ID・parent・restored_from・復元先文書全体の
一致を要求する。旧runtimeがerrorを記録したrestore journalも起動時に再照合する。
公開状態を読めない場合は成功扱いせず処理を止める。exportのrename/fsyncエラーも、
manifestと全公開bytesが照合できた場合だけ成功として返す。

## ブラウザでの閲覧と編集

`start` / `serve` に `--ui explain_html.web:files` を付けると固定の閲覧・編集UIを使える。
fragmentを含む起動URLを開く。UIはfragmentを直ちに除去し、tokenをメモリだけに保持して
認証が必要なJSON APIにだけ付ける。ページ全体の再読込には起動URLが再度必要になる。
進行中のrequest ID・操作種別・base・開始時刻だけを同じタブのsessionStorageへ残し、tokenや
指示本文は保存しない。切断復旧は保存IDで照合し、編集・undoを自動再送しない。
undoの応答が失われても、履歴への未掲載は未確定として扱う。要求IDを保持して公開履歴の照合を
続け、表示版を読めるまま既存の「状態を再確認」操作も提供する。

対象メニューで全体または内容blockを指定する。本文の見出し・図のクリック、Enter/Space、
同じblock内の文章選択でも対象を選べる。選択quoteとUTF-16位置は表示版に照合して送信する。
固定UIに指示・cancel・経過時間・公開エラー・版履歴・過去版の閲覧・復元を置き、回答は本文へ
反映する。再試行は明示操作で新request IDと表示中の現在版baseを使う。古いrevisionの競合では
本文を保持し、明示操作で現在版へ更新できる。

`GET /api/documents/{document_id}/revisions/{revision_id}/content/reader.html` は公開済み版の
閲覧専用view。opaque iframeから固定UIへDOMを公開できず、srcdocは固定UIのCSPを継承するため、
serviceへこの局所拡張が必要になる。最小の閲覧支援styleと固定の `web/reader.js` を加え、その正確なbytesのscript
hashを既存rendererのhashと一緒に許可する。この一時responseだけはCSP metaをHTTP CSPで置き換え、
sandbox・connect-src none・resource制限・renderer hashを維持する。共通のcontent file_name列挙にもreader.htmlを追加するが、JSONモデルとexportのファイル一覧は維持する。保存HTML・validator・compiler・
`content/index.html`・exportのbytesは変更しない。

iframeは `sandbox="allow-scripts"` のopaque originで、same-origin・forms・popups・top-navigationの
権限を持たず、token/cookieも渡さない。生成scriptは引き続き拒否する。固定UIが受け取るmessageは
同じframe window・origin `null`・表示document/revision・viewごとのnonceに一致する `ready`・
`selection`・閲覧 `state` だけ。内容ID/quote/位置とstateの形を表示版に照合し、指示送信には使わない。
閲覧viewの解析には共通のStrictHTMLParserを使う。対象索引はinertなtemplateで作り、解析したnodeを
固定UIのDOMへ入れない。この解析時にはapp.jsからinline style属性のCSP拒否が報告され得る。
固定UIのpolicyは緩めず、本文styleはiframe内だけへ適用する。
親からiframeへのmessageはopaque originのためtarget origin `*` を使い、接続・状態復元/印・移動の
データだけを含む。bridgeは親windowとHTTP originを確認し、tokenの読取りやservice通信を行わない。

処理中も旧frameを読める。成功した公開版は別frameで準備し、bridgeの初期化後に切り替える。
HTML・図定義・参照asset metadataが同じblockでは、block自身がdetailsの場合も展開状態とグラフ入力値を保持する。
読み位置は画面上端を含む最も内側の可視blockの内容IDと相対位置で復元し、外側のページをfocus/scrollしない。変更は控えめな内側の印で示し、
変更箇所への移動は明示ボタンだけで行う。読んでいた節が消えた場合は通知し、元のscroll高さを
新文書の範囲に収める。対象が削除・変更された場合は、送信前に選び直してもらう。

過去版のpreviewは、現在版へ戻るか新revisionとして復元するまで編集不可。ZIP exportにはserviceの
正確なsnapshot一式を入れる。操作UI・bridge・認証・要求記録を含まず、展開後のindex.htmlから静的な
説明・画像・対応グラフをofflineで読める。AI画像の生成は初版だけで、閲覧中はSVG・グラフ・既存画像を
使える。新規画像の要求はimage_generation_unavailableとして理由付き未完了を表示する。

## 保存・復旧・export

ソース内使用例の `outputs/` のように、Git対象外の専用runtime directoryを使う。root全体は
非公開にし、公開は許可リストAPI経由に限定する。保存配置:

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

各revisionはHTML・根拠・図定義・asset bytesの完全な組を持つ。全ファイルとdirectoryをflushしてから
`current.json` を原子的に置き換える。それ以前の失敗では現在版を保つ。置換境界のprocess/power lossでは
旧版または新版の完全な組になり、応答が不確実なら再openしてrequest IDで照合する。
失敗したpending directoryと到達不能の保存済み版は非公開に残り、履歴や `revision()` へ出ない。
現在版の破損時に古い版を黙って選び直さない。background GCはない。

`current()`・`revision()`・`history()` は公開済み版を返し、履歴は新しい順。同じstaged candidateの
成功後commitは冪等で、同じrequest IDの別内容は競合する。`restore()` は旧版の完全な文書を
**新しいrevision**へコピーし、現在版をparent、復元元を `restored_from` として記録する。
全bytesを再検査し、同じ旧文書の合格済みsemantic/rendering判定を再利用する。古いbase IDが
現在版へ戻ることはない。同一restoreの再送は作成済みrevisionを返す。

`snapshot()` は明示した公開済みrevisionを固定し、独立したbundleを原子的に設置する。
公開ファイルはHTML・申告済みassets・根拠・図・公開manifestに限定する。storeのtree、要求、
検証結果、raw prompt、認証、privateログをコピーしない。根拠と由来を含め、Documentの全情報は
公開用。非公開文脈をDocumentへ入れない。`asset_bytes()` はmanifestのID、`snapshot_file()` は
snapshotの許可リストから解決する。絶対・親逸脱・encoded path、symlinkの各component、hard linkを
拒否する。asset pathは `assets/<asset_id>.<ext>` に統一する。対象はWSLを含むlocal POSIX filesystem。
Windows Python、network filesystemの原子性、実power lossの耐久性は本子では実証していない。

## 図と画像生成

定性的な場面・比喩には生成画像、正確な構造にはSVG、数値比較・パラメータ操作にはmanaged graphを
使う。すべてのAsset/FigureがID・caption・alt・公開用の由来を持つ。managed markupはcaption・
由来の種別・出典を図の近くへ表示し、画像のaltとグラフのaccessible labelにはFigureの値を使う。
文脈に応じた図のcaptionは再利用assetのcaptionと異なってよいが、画像/SVGの由来は一致させる。

`images.import_image(path, asset_id=..., caption=..., alt=..., provenance=...)` は
`.asset`と未加工の`.content`を持つ`ImportedAsset`を返す。decodeできる単一frameのPNG/JPEG/WebPに
限定し、16 MiB・16 megapixelを上限とする。対話Codex imagegenで保存した画像も同じ入口でimportし、
由来を`generated_concept`にする。`visuals.import_svg(content, ...)`はSVG namespaceと有限で正の
viewBoxを持つ、サイズ制限付きの幾何・文字を受け付ける。script・foreignObject・animation・style・
entity・resource参照を拒否し、黙って削除しない。SVGはimgとして埋め込み、objectやactive inline
markupにはしない。

`managed_renderer()`のAsset/bytesとRenderer version `2`を使い、graphのFigure.asset_idsはこの
rendererだけにする。Figure.definitionは`parse_graph()`で検査する。既存wire契約内のgraph設定の
正本はmodelとその`model_json_schema()`。未知field・数値の型変換・非有限値・軸範囲外dataは拒否する。
対応typeは`observation_error`と`comparison`だけで、式・JSのeditorはない。前者は明示的な仮想例で、
rectification済みの校正されたpinhole stereoと正しい対応を仮定し、focal/disparityのpx、baselineのm
から`Z = f B / (d + error)`でmの深度を求める。符号付き誤差slider・sample入力・軸の単位/範囲・表示dataを
一致させる。比較図は、増加するx値と単位/範囲が明記された最大8系列を保持する。
[具体例fixture](../../../tests/support/agents_ops/explain_html_visuals.py)を参照する。

任意の配置へ`<figure data-figure-id="..."></figure>`という空slotを置き、宣言したcontent blockの内側
またはblock自体にする。head/bodyと明示的な閉じタグが必要。`render_document(document)`が図・CSP meta・
正確なmanaged scriptを埋め込むので、**候補の変更一覧を計算する前**に実行する。graphには初期SVG・
元data表・仮定・単位・入力範囲・accessibleな操作を含め、JS無効でも初期説明を読める。同じscript bytesを
Assetへ保存しHTMLにも埋め、CSP hashで許可する。scriptは最初のgraphのcontent blockに置き、DOMの
解析完了後に全graphを初期化して参照の所有関係を保つ。file://における外部scriptのSRI/CORS失敗を回避し、
snapshot後にCDN・fetch・module import・serverを必要としない。CSPの挿入位置はbodyより前にある
一意のheadを解析して得る。comment中のheadという文字列には影響されず、head内の最初のscript・style・
resource参照より前へ配置する。不正なhead/bodyの入れ子と、head前の非ASCII空白（実体参照のNBSPを含む）は拒否する。

CSP・renderer script・`data-visual-style`の図用CSSはcompilerが所有する。
writerには編集時も省略するよう明示し、コードやhashの複写を要求しない。
同等のCSP metaはdirective名とtoken集合が必須policyと一致する場合だけ受け付け、正本へ置換する。
属性順・引用符・http-equivの大文字小文字・directive順・空白の違いを許容する一方、
directive/policy重複、追加directive、参照先変更、弱いpolicyは拒否する。
既存の完全一致するmanaged scriptは置換し、コード改変・追加属性・任意scriptは拒否する。
既存の管理CSSも正本との一致を要求する。これは意味検査より前に行う明示的なcompile契約で、
静かな補修ではない。stage後は正本との完全一致を引き続き要求する。
管理CSSは`table{min-width:750px}`のようなwriterのglobal指定があっても、390px幅で
元data表と図をfigure内に収め、横scrollさせる。

数式はChromium標準のpresentation MathMLで表現し、CDN・依存追加は不要。
writer promptとvisual許可リストの要素を揃え、annotation-xml・mglyph・link・異種要素・event属性は拒否する。
TeXのcommand/区切りを拒否し、未描画の数式を公開しない。平易な説明を添え、静的snapshotにも同じbytesを保存する。

TeXの文字列検査はfigure captionを含む本文が対象。`$...$`・`$$...$$`・`\(...\)`・`\[...\]`と、
バックスラッシュ直後にASCII英字が続く命令全般を拒否する。text run内ではinline要素・comment・
展開済み文字参照をまたいで文字列を連結し、命令や区切りの分割も検出する。
block要素とliteral領域でrunを区切り、別の段落同士は連結しない。
code/pre/kbd/samp/mathの子孫、属性値、headのmetadataはこの本文検査から除外する。
ソース記法やpathはliteral要素に置く。除外範囲でもMathML・HTML/CSS/script/属性の安全性検査は維持し、
余分なmarkupを許可しない。対応のない単一ドルはliteralとして扱う。
隣接する2つのドル記号がどちらも数値価格を始める`$5 and $10.50`などは通貨と判定する。
金額は符号・小数/桁区切りを許し、数値の直後が空白・句読点・run末尾であることを要求する
（次の価格直前のハイフンも区切り）。`$5$`・`$x$2`・価格の後の数値数式は拒否する。
曖昧なliteral記法はliteral要素へ置く。違反はcompile・stage後のvisual検査で`invalid_document`とし、
本文を書き換えたり旧判定を使い回したりしない。TeX専用の自動retryは追加しない。

描画属性（`data-managed-*`・`data-definition`・`data-plot`・`data-values`・`data-slider-value`・
`data-render-*`）は正規graph figure内へ予約し、`data-graph-targets`はmanaged scriptだけが持つ。
v2はscript内のfigure/definition一覧を読み、宣言したfigure slotの直下要素だけを対象に設定を照合する。
余分な一致要素をgraph入力として探索しない。壊れた対象は明示的な図のエラーにし、ほかの図の初期化は続ける。

v1のbytesは変更しない。`managed_renderer("1")`で旧Asset/Figureをsnapshot/復元でき、既定はv2とする。
1文書ではrendererを1版に揃え、混在は明示的に拒否する。将来の実装は新file/ID/versionを使い、旧版の
HTML/assetsはそのrevisionから提供して現在のcodeを注入しない。runtime検査は追加防御であり、公開前検査と
iframe隔離を置き換えない。

stage後、store APIから得た不変の文書と**全staged asset bytes**を使い、
`validate_visuals(document, all_asset_bytes_by_id)`を呼ぶ。参照の閉包・digest・MIME・asset内部・
managed scriptの同一性・正規の図/data markupを確認する。inline event handler・任意script・
埋め込み閲覧context・URLを含むCSSは拒否する。通常のCSSとmedia/supports/layer/container規則は許可するが、
import・escape・CSS commentはこの保守的なsubsetの対象外。これは意味/表示の合格判定を生成するものではなく、
信頼するcontrollerはcommit前に独立した意味/ブラウザ検査を行い、編集credentialのないiframe sandboxで
表示する必要がある。`content_security_policy(document)`をHTTP headerへも設定できる。

AI生成画像は**初版の作成時だけ**対話Codex imagegenで生成してローカルimportする。閲覧中のcontrollerは
`UnavailableImageGenerator`を使い、`capability()`は初版限定という理由付きの利用不可を返す。
閲覧中の新規AI画像要求は共通エラー`image_generation_unavailable`にし、**画像要求全体を未完了**として
公開を中止し、現在版を保持する。部分的な文章/SVGへの代替を画像生成成功として公開しない。
既存画像のimport・再配置/caption変更・SVG/graph更新は独立した要求として利用できる。
これは[確定した人間の判断](https://github.com/Motoki0705/tennis-lab/issues/1710#issuecomment-6732169335)で、
本範囲にservice adapter選択の未決事項はない。

`generate_image(..., adapter=...)`は検証済みの`ImageGenerator`境界（能力照会と実bytes/公開sourceを返す生成）を
保持する。注入adapterは成功/不可/失敗の契約検証に使い、閲覧serviceは生成providerを選択しない。
provider/不正成果物の失敗は`asset_generation_failed`とし、providerの内部情報を公開せず、同様に
要求全体を未完了として現在版を保持する。認証やproviderを自動探索しない。対話imagegenの存在を
local serviceの生成能力とは扱わず、能力調査の根拠はPRへ残す。

## 下流の責務

公開ファイルを直接書き換えず上記runtimeのHTTP/公開境界を使う。storeのlock/base照合は
信頼する外部writerとの競合も防ぐ。UIはsandbox表示・選択messageの厳格な検査・変更箇所移動・
静的exportの編集不可表示を担当する。本文と全asset URLを同じrevisionへ固定し、jobがsucceededとして
新revisionを返した後に更新する。

## 検証

repoの `.venv/bin/python -m pytest` で `tests/unit/agents_ops/explain_html` と
`tests/integration/agents_ops/explain_html` を実行する。件数は `--collect-only`、時間は
`--durations=0 --durations-min=0` で測る。小さい実ファイル、
非対称ブロック、複数assets、宣言的グラフデータ、例外注入、同期した競合writerを使う。
外部API・GPU・browser・実時間sleepは不要。visual追加分はPython unit 9件・Node 3件で、Nodeは
`node --test tests/unit/agents_ops/explain_html/visuals.test.mjs`で実行する。
既存の復元/snapshot integration 2件は実画像/SVGとmanaged graphを使い、追加のintegration 1件で
v2公開後のv1 snapshot/復元と元のv1 bytesを検証する。ブラウザ検証はChromiumのsetupを共有する2経路:
offline slider/比較図/CSP配置と、不正構文/SVG/予約属性のexport拒否・現行版保持・対象別の失敗隔離・
JS無効時の初期表示・旧v1の再表示。

```sh
PLAYWRIGHT_MODULE=/path/to/playwright CHROMIUM_PATH=/path/to/chromium \
  node tests/e2e/agents_ops/explain_html_visuals_browser.cjs
```

harnessはsetup/各経路の時間と、一時screenshot directoryを出力する。
通常テスト/CIで実画像生成や外部APIを呼ばない。
既存のJavaScript pytest入口・CIレーンからもvisualの両suiteを実行する。

editorは `test_editor.py` にintegration 11件、
`tests/e2e/agents_ops/test_explain_html_editor.py` に公開入口3件を追加する。
偽agy実行ファイル・実store・長文の完全な入力・許可/禁止した根拠変更・拒否/不正verdict・
実hookの拒否・pipe/Event同期によるprocess回収を検査する。1件は実際の `render_document` と
stage後の `validate_visuals` をeditorの意味検査・store commitへ接続する。
editorの検証にはbrowser・固定の実時間sleepは不要。

runtimeは `test_jobs.py` の実store integration 17件と、
`tests/e2e/agents_ops/explain_html/test_service.py` のHTTP/CLI 6件を追加する。
偽editor/描画判定、Event gate、OS割当port、privateな一時rootを使い、生成/commitの重複・要求単位のcompile bytes・古いbase・
cancel/undo/shutdown後の遅延結果・進行中journalの復旧・現在版切替後のdiskエラー・session間のtoken分離・
Host/Origin/path拒否・公開export範囲・CLIの起動/終了/再起動/offline exportを検査する。
restoreはpointer置換前/後の障害・公開状態を読めない場合・旧error journalを検証する。
storeの追加3件はexportのrename障害とreadback失敗を扱い、実directory fsync呼出しへ障害を注入する。

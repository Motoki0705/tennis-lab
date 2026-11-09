# 説明HTMLの文書契約

[English](README.md) · [简体中文](README.zh-CN.md)

[#1710](https://github.com/Motoki0705/tennis-lab/issues/1710) の文書・編集の共通契約と、
永続的な版store、visual import・managed renderer・検証済み画像adapterの境界を提供する。
Gemini呼出、HTTPサーバ、ブラウザUI、画像provider実装、品質評価は含まない。
既存依存のPydantic/PillowとPython標準ライブラリを使う。

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
| [images.py](images.py) | ローカル画像import、能力の報告、失敗時に停止する画像adapter |
| [visual_definitions.py](visual_definitions.py) | 観測誤差例・比較グラフの厳密な宣言的設定と数値モデル |
| [visuals.py](visuals.py) | SVG/HTML/CSSの安全検査、図のmarkup、offline renderer policy |
| [renderers/visuals-v1.js](renderers/visuals-v1.js) | 実装を固定したmanaged renderer version 1 |

## ID・対象・候補

生成側は編集対象となる論理ブロックへ `data-content-id` を付ける。同じ論理内容のIDを保持し、
新しい内容には新IDを付ける。見出し・配置・配色・ブロックのタグ・順番は固定しない。
HTMLは明示的で対応の取れた閉じタグを使う。HTML ID・内容ID・成果物IDの重複は失敗する。
HTML本来の `id` と `data-content-id` は別の名前空間。図・根拠は `data-figure-id`・
`data-evidence-id` で参照し、リソースには `data-asset-id` を使える。

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

`managed_renderer()`のAsset/bytesとRenderer version `1`を使い、graphのFigure.asset_idsはこの
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
Assetへ保存しHTMLにも埋め、CSP hashで許可する。file://における外部scriptのSRI/CORS失敗を回避し、
snapshot後にCDN・fetch・module import・serverを必要としない。version 1は変更せず、将来のrendererは
新file/ID/versionにする。旧版のHTML/assetsはそのrevisionから提供し、現在のcodeを注入しない。

stage後、store APIから得た不変の文書と**全staged asset bytes**を使い、
`validate_visuals(document, all_asset_bytes_by_id)`を呼ぶ。参照の閉包・digest・MIME・asset内部・
managed scriptの同一性・正規の図/data markupを確認する。inline event handler・任意script・
埋め込み閲覧context・URLを含むCSSは拒否する。通常のCSSとmedia/supports/layer/container規則は許可するが、
import・escape・CSS commentはこの保守的なsubsetの対象外。これは意味/表示の合格判定を生成するものではなく、
信頼するcontrollerはcommit前に独立した意味/ブラウザ検査を行い、編集credentialのないiframe sandboxで
表示する必要がある。`content_security_policy(document)`をHTTP headerへも設定できる。

`generate_image(..., adapter=...)`は実際に得た検証済み画像だけを返す。既定の
`UnavailableImageGenerator.capability()`は理由付きの利用不可を返し、認証やproviderを自動探索しない。
信頼する別途検証済みadapterは`ImageGenerator`（能力照会と、bytes/公開sourceを返す生成）を実装する。
利用不可は共通エラー`image_generation_unavailable`、provider/不正成果物の失敗は
`asset_generation_failed`にし、providerの内部情報を公開しない。controllerはどちらでも**画像要求全体を
未完了**として公開を中止し、現在版を保持する。部分的な文章/SVGへの代替を画像生成成功として公開しない。
既存画像のimport・再配置/caption変更・SVG/graph更新は独立した要求として利用できる。対話imagegenの
存在はlocal service adapterの証拠にはならず、serviceからの実行・認証・成果物回収を別途probeする。
能力調査の根拠と人間判断の未決事項はissue/PRへ残す。

## 下流の責務

HTTPのJSON入口は `contract_schema()` に定めたsession-token headerを使う。
隔離する本文/exportへ編集credentialを渡さない。routeとbodyのIDを一致させ、内容ファイルとasset名を
選択したrevisionの許可リストへ照合する。任意のHTTP pathをprivate treeへ連結しない。
HTML/assetsの取得前にrevision IDを固定して版の混在を防ぐ。公開エラーはschemaの形式だけとし、
生の例外を返さない。

job controllerは生成実行前の永続的なrequest重複排除、cancelとcommitの同期、job遷移、
interrupted復旧を担当する。commit時のcancel/revision再確認、実行中の旧版保持、状態取得や再接続での
再生成禁止が必要。storeのlock/base照合は競合公開を防ぐがprocessやcancel状態は管理しない。
UI/rendererがsandbox、選択通知、変更箇所移動、静的exportの編集不可表示を担当する。

## 検証

repoの `.venv/bin/python -m pytest` で `tests/unit/agents_ops/explain_html` と
`tests/integration/agents_ops/explain_html` を実行する。件数は `--collect-only`、時間は
`--durations=0 --durations-min=0` で測る。小さい実ファイル、
非対称ブロック、複数assets、宣言的グラフデータ、例外注入、同期した競合writerを使う。
外部API・GPU・browser・実時間sleepは不要。visual追加分はPython unit 9件・Node 3件で、Nodeは
`node --test tests/unit/agents_ops/explain_html/visuals.test.mjs`で実行する。
既存の復元/snapshot integration 2件は実画像/SVGとmanaged graphを使う。ブラウザ検証はChromiumの
setupを共有する2経路（offline slider/比較図と、不正SVGの拒否・現行版保持・JS無効時の初期表示）:

```sh
PLAYWRIGHT_MODULE=/path/to/playwright CHROMIUM_PATH=/path/to/chromium \
  node tests/e2e/agents_ops/explain_html_visuals_browser.cjs
```

harnessはsetup/各経路の時間と、一時screenshot directoryを出力する。
通常テスト/CIで実画像生成や外部APIを呼ばない。
既存のJavaScript pytest入口・CIレーンからもvisualの両suiteを実行する。

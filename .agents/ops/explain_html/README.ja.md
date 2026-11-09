# 説明HTMLの文書契約

[English](README.md) · [简体中文](README.zh-CN.md)

[#1710](https://github.com/Motoki0705/tennis-lab/issues/1710) の文書・編集の共通契約と、
永続的な版storeを提供する。Gemini呼出、HTTPサーバ、ブラウザUI、画像生成、品質評価は含まない。
既存依存のPydanticとPython標準ライブラリを使う。

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
外部API・GPU・browser・実時間sleepは不要。

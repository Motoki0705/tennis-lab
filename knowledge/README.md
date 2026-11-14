# Knowledge Control — 研究ライブラリ

実験の結果・考察・関連研究をgitで共有する。**1 run = 1ノード**。実験群はgroupで束ね、前提・比較・反証の関係はIDで繋ぐ。
現在の判断は [summary.md](summary.md)、閲覧は [Web UI](webui/README.md)、登録・更新・検索の手順は [knowledge-control skill](../.agents/skills/knowledge-control/SKILL.md) を起点にする。このREADMEが保存形式の正本。

実RGB SLCSの教師・学習結果・採否の総括は [結果レポート](reports/slcs-real-rgb.md) を参照。

## 保存構造

```text
knowledge/
  README.md
  summary.md                         # タスクごとの現在の判断。詳細の正本は各ノード
  history/summary-2026-11-05.md      # 2026-11-05までのsummary本文の凍結アーカイブ（追記しない）
  registry/                          # タスク・関係の語彙と、標準pipelineのcheckpoint対応表
    tasks.yaml
    relations.yaml
    deployments.yaml
  nodes/
    plcs/.sequence                  # 過去に予約した最大番号（削除後も保持）
    plcs/000001-run-<slug>.md
    blcs/000001-run-<slug>.md
    synthetic_data_generation/...
    <task>/000002-group-<slug>.md
  Papers/
    paper-2024-gvhmr/
      paper.md                       # 人間が採録した論文の書誌・要約・採録理由・読解
  citation_map/                      # 探索した文献。Papersへの登録とは独立
    papers/cite-<identity digest>.md  # 文献・参考文献ごとの判断（未読の参照先も含む）
    searches/search-<slug>.md         # 探索条件と世代・問題箇所への参照
  reports/                           # 複数ノードにまたがる結果レポート
  runs/<run-id>/                     # 小さな再現性bundle・指標・図・削除記録
  run-storage-exceptions.json        # 既存bundleの固定例外（自動追加しない）
  webui/
```

タスクは小文字snake_caseの**拡張可能なトピック**。`plcs` / `blcs` / `court_detection` / `ball_detection` に限定しない。合成データ生成、統合、独立した研究クラスタも追加できる。新しいトピックは[タスク登録簿](#レジストリ)に1件追加してから最初のノードを登録する。ノードは主目的となる1タスクに置き、タスク横断の関係はparents / relations / membersで表す。同じノード・論文を複製しない。issueは分類階層ではなく検索用メタデータ。

### ID・連番・ファイル名

- 不変ID: runは `run-<slug>`、groupは `group-<slug>`。slugは小文字英数字をハイフンで繋ぐ。新規は `run-i<issue>-<短い施策名>-s<seed>` など、差分がわかる名前にする。issueやseedが無い場合は省略し、架空の値を付けない。
- ファイル名: **`<6桁連番>-<id>.md`**。タスク内でrun/group共通の連番を1から付ける。IDと連番は別物で、関係参照は連番やファイル名ではなくIDを使う。
- `sequence` は登録順。新規登録は `.sequence` に記録された予約済み最大値+1、既存の番号は変更・再利用しない。`--force`も番号とタスクを維持する。削除や登録中断で欠番が生じても詰めない。`.sequence` はノード削除後・タスク廃止後もgitに残し、削除・縮小しない。欠落や現存ノードより小さい値は検証エラーにする。
- `date` は実験日（判明している場合のみ）、`recorded_at` は登録日。過去実験の後日登録は末尾に追加し、UIの時系列表示は `date` 優先で並べる。
- 既存200ノードの初回移行では実験日→ID順で採番。日付不明のノードはGit追加日を使用し `date_source: git_added` を記録した。判明しているものは `experiment_date`。同日内の実際の実行順は復元できないためIDで決定した。初回の `recorded_at` はこの移行時のソート基準日。
- 同じcheckout内の同時登録はディレクトリロックで直列化する。独立したworktree/branchはそれぞれ採番するため、マージ時の重複番号はvalidatorがエラーにする。**未マージ側の新規ノードだけ**をマージ先の最大値以降に採番し直し、`.sequence` は両branchの予約済み最大値と新規番号以上に合わせ、リンクとsummaryを更新する。ロックが別branchまで一意性を保証するとは扱わない。

## ノードのfrontmatter

```yaml
---
id: run-i900-geometry-s42
type: run
task: plcs
sequence: 103
recorded_at: '2026-09-20'
title: 幾何特徴を用いた位置推定
issue: 900                         # 整数または整数配列
provider: codex                    # claude / codex / gemini / human / other
date: '2026-09-20'                 # 実験日が不明なら省略
status: done                      # done / failed / running / planned
summary_impact: none               # updated / none（追加・status変更したノードで必須）
config: {model: multiview_axial_base, run.seed: 42}
metrics: {position_error_m: 0.38}   # 実測値のみ。split・単位を本文に記す
repro: {commit: '<sha>', command: '<exact command>'}
artifacts:
  run_dir: knowledge/runs/run-i900-geometry-s42
parents: []                        # baseline / 前提 → このノード
relations: []                      # [{to: run-..., rel: compares, note: 任意の補足}]
papers: [paper-2024-gvhmr]         # 論文ID。本文で関係を説明
tags: []
---
```

`id` / `type` / `task` / `sequence` / `recorded_at` / `title` が必須。titleは空白だけでない文字列。
`session`、`repro.branch` / `remote`、`artifacts.predictions` / `curves` / `log` / `output_dir` / `tb_logdir` は利用可能な根拠を記す。
`parents`, `members`, `tags`, `papers` は空でない文字列の配列（重複不可、配列自体は空でも可）。`relations` は空でない文字列の `to` と `rel`、任意の `note` だけを持つmappingの配列。これらは未使用なら省略できる。

`config`, `metrics`, `repro`, `artifacts` はmapping。文字列キー、JSON互換の値（数値・真偽・文字列・null・配列・入れ子mapping）を使う。NaN/InfinityやYAML固有オブジェクトは不可。未測定は省略・空mapping・null値とし、理由を本文に記す。任意の `issue` は正の整数またはその非空配列。`status` / `provider` は上記の値を使い、未確認なら省略またはnull。`date` / `recorded_at` は実在する `YYYY-MM-DD`（引用推奨）。`date_source` は移行由来の `experiment_date` / `git_added` のみで、前者には `date` が必要。

frontmatterはYAML mappingとして記述し、重複キー・alias/mergeを使わない。設定のコピーが必要なら値を明示する。登録スクリプトの出力は雛形であり、見出しとコメントだけの本文はCIエラーになる。考察の具体的な見出し・順序は強制しない。

- **group**: `type: group`、`id: group-...` とし、`members` に既存run/groupのIDを列挙する。本文は群の結論を書く。
- **parents**: baseline / 前提から子への有向関係。循環不可。groupの `members` 入れ子も循環不可。すべての関係で自己参照は禁止。
- **relations**: 非階層の有向関係。`rel` は[関係の語彙](registry/relations.yaml)から選び、補足は `note` に書く（`related` はnote必須）。
- **papers**: 関連研究の出典。引用だけで再現・実証したとはみなさない。背景、実装採用、比較対象、仮説のどれかを本文に説明する。

- **summary_impact**: summary.mdとの照合結果。`updated` はこの変更でsummaryの判断を書き換えたこと、`none` は判断を変えないことを示し、理由は本文に書く。ブランチで追加したノードと `status` を変えたノードに必須（[summaryと履歴](#summaryと履歴)）。

runに `members` は指定しない。groupには1件以上のmemberが必要。関係の種類を増やすときは `registry/relations.yaml` を改訂する。比較関係同士の循環は許可する。

`artifacts.run_dir` を記録した場合は、そのノード自身の `knowledge/runs/<id>/` ディレクトリが必須。通常はrepo相対パスを使う。
ローカルの場所はcheckout相対（メインcheckoutの `.training_queue/` や `.claude/worktrees/` も含む）、それ以外のホーム配下は `~/` で記録し、Drive上の場所はDrive root相対の `drive_record`（Colabの運用記録）・`drive_outputs`（出力ディレクトリ）で記録する。新規ノードのartifactsにホームディレクトリの絶対パスを書くとエラーにする。登録スクリプトはこの形へ変換する。`KNOWLEDGE_DIR` で隔離したライブラリを使う場合はそのライブラリの `runs/<id>` を指す。`log` / `output_dir` / checkpoint等のローカルパスはCIでの存在を要求しない。
`runs/` の再現コマンド・`repro.sh`・`run.json`・patchは歴史的証拠なので、保存基準の変更やノード移動に伴って書き換えない。

## レジストリ

`registry/` の3ファイルは、ノード以外の語彙と対応表の正本。どれも必須で、欠落・不正はエラーにする。ファイル内のパスはライブラリの親ディレクトリ（repo root）からの相対パス。

| ファイル | 内容 | 検査 |
|---|---|---|
| `tasks.yaml` | トピックごとの `status`（`active` / `retired`）、`module`、`note`、`successor`、`since` | `nodes/` の全ディレクトリと論文の `tasks` が登録済みで、`module` が実在する。moduleの無いトピックと廃止したトピックは `note` 必須。廃止したトピックには新しいノードを登録できない |
| `relations.yaml` | `relations[].rel` の語彙と説明。`note_required: true` の関係はnote必須 | 未知の `rel`、`to` / `rel` / `note` 以外のキー |
| `deployments.yaml` | `configs` に挙げたconfigが選ぶcheckpointごとの採用根拠。学習したモデルは `evidence`（ノードID）、外部の公開重みは `source`（重みの出典を記した文書）、任意で `papers` / `since` / `note` | configの `checkpoint` / `*_checkpoint` キーと登録簿が1対1で、値も一致する。根拠ノード・論文・出典文書が実在し、出典文書がファイル名に言及する |

標準pipelineのcheckpointを変えるPRは、`deployments.yaml`・根拠ノード・summary.mdを同時に更新する。CIは追跡するconfigと `LICENSES.md` の変更でも動く。

## run成果物の保存基準

この節の容量・種別の制約は **`runs/` に適用する**。文書ライブラリ、reports、Web UIには一律に適用しない。
Gitに残すものは、研究の考察・指標・設定・正確なコマンド・環境情報・再現スクリプト・小さな図。
通常ファイルは **256 KiB以下**、図（PNG・SVG・PDF）は **512 KiB以下**、run合計（削除記録を含む）は **2 MiB以下**。KiB=1,024 bytes、MiB=1,048,576 bytes。境界値ちょうどは許可する。

- 予測配列（NPZ・NPY）、モデル重み、TensorBoard event、圧縮ファイル・ログ、動画・GIF・JPEG、プロファイルはサイズにかかわらず置かない。
- 許可するテキスト系の拡張子は `.py .sh .yaml .yml .json .csv .jsonl .txt .log .md .patch .cjs .state .job .ps1 .toml .js .mjs .html .css .ts`。上記の図以外の種別とsymlinkはエラー。新たな種別を扱うときは基準と検査器を明示的に改訂する。
- `kg_register.py` は再現bundleと小さな指標JSONだけをコピーする。予測配列はコピーせず元に残し、その旨を表示する。上限を超えるbundleはコピー前に拒否する。再現用patchも新規登録では通常上限に従う。
- 既存の大型パッチ34件と、初回整理後も合計上限を超える13 bundleは [固定例外](run-storage-exceptions.json) にパス・SHA-256・理由を記録した。ファイル例外は内容、run例外は全相対パスと内容の集合を固定する。runのhashは、相対パス順に並べた `<ファイルSHA-256>  <run相対パス>\n` のUTF-8バイト列のSHA-256。追加・変更・不要になった例外は検査エラーで、登録処理が例外を広げることはない。
- 初回整理前からある別用途の `artifact-manifest.json` 1件（比較bundleの内容目録）は、同じ一覧の `legacy_manifests` でパスとhashを固定して原文を維持する。削除記録と混同せず、未登録の形式を自動的に許可しない。

### 削除した成果物の記録

2026-10-19の初回整理は、承認された基準で該当成果物を**保存先を作らず削除**した。記録は内容を復元するための保存先ではない。履歴からの除去後は現物を再取得できない。
各runの `artifact-manifest.json` が削除記録の正本。全ファイルが対象のrunも、この記録によりディレクトリを残す。

```json
{
  "schema_version": 1,
  "run_id": "run-example",
  "files": [
    {
      "path": "predictions/pred_test.npz",
      "size_bytes": 1234,
      "sha256": "<元ファイルのSHA-256、64桁小文字hex>",
      "reason": "raw-data-events-archives",
      "removed_from_commit": "<削除直前のHEAD、40桁commit SHA>"
    }
  ]
}
```

`path` はrunディレクトリからの元の相対パス。`removed_from_commit` は削除時に観測した**変更前のHEAD**を指す（記録を含むcommit自身は自己参照しない）。保存先のフィールドは作らない。
削除理由は `raw-data-events-archives` / `figure-over-512KiB` / `table-or-log-over-256KiB` / `file-over-256KiB` / `unsupported-file-type` / `run-over-2MiB`。
形式はversion 1だけを受け付け、未知の版・フィールド・重複パス・不正なhash・種別/容量と矛盾する理由・現物が残る記録を拒否する。
`files` の順序と既存項目は変更しない。後の削除は末尾に追記し、JSON Pointerを維持する。

削除したartifactは、元の参照を次のmappingに置き換える。本文リンクは `[元の説明（削除済み）](../../runs/run-example/artifact-manifest.json#/files/0)` とする。画像も削除記録への通常リンクに変える。研究内容・指標・過去の再現コマンドは変更しない。

```yaml
artifacts:
  run_dir: knowledge/runs/run-example
  predictions:
    status: deleted
    record: knowledge/runs/run-example/artifact-manifest.json#/files/0
```

全内容を削除したディレクトリへの参照は、`status: deleted` とmanifest全体の `record` に、元のrun相対ディレクトリを示す `path_prefix: predictions/` を加える。対応する削除記録が1件以上あり、そのディレクトリが残っていないことを検査する。

検査器はrun配下へのartifact・本文リンク・本文の `knowledge/runs/...` 参照と、記録のJSON Pointerを検査する。ローカルの `output_dir`・checkpoint・ログ等は従来通りCIでの存在を要求しない。歴史的bundle内部の入出力参照は変更せず、実行scriptの可用性は `kg_repro_paths.py` が検査する。
`--base-ref` では、新しい削除と記録の1対1対応、削除前のGit blobと記録のサイズ・SHA-256、変更前commitにそのblobがあること、既存記録の不変性も検査する。履歴書き換え後も、既存記録の検査に旧commitの取得を要求しない。

履歴から除くためのHEAD削除パス一覧は、検証した記録から機械的に取り出せる（標準出力はソート済みのrepo相対パスだけ）。

```bash
mkdir -p tmp
.venv/bin/python .agents/skills/knowledge-control/scripts/kg_artifacts.py --removed-paths > tmp/knowledge-removed-paths.txt
```

この一覧は記録済みのHEAD削除分だけを表す。過去の履歴にだけ存在した対象パスは、履歴整理の実施時に別途調査する。履歴書き換えとforce pushはHEAD整理とは別工程で行う。

### 履歴書き換え後のcommit参照

`kg_remap_commits.py` はfilter-repoの `commit-map` から、追跡中のテキストにある完全SHAと文脈を確認できる7桁以上の短縮SHAの更新案を作る。新SHAはbranch・tag・remote-tracking branchから到達可能なものに限る。新しいcheckoutで実行し、差分を通常のcommit/PRとして適用する。履歴自体や外部サービスは変更しない。

短縮SHAの文脈にはcommit/revision等の項目、引用・Markdown強調で囲んだ英字を含むhex文字列、`Recorded from …`、このrepoのcommit/tree/blob/compareへのリンク（表示部分を含む）を使う。日本語に隣接したSHAも検出する。commitの文脈がない数字だけの引用や浮動小数点数、ファイル名・実行IDに埋め込まれた短縮形は自動更新せず、監査時に実際のcommit参照か確認する。

```bash
.venv/bin/python .agents/skills/knowledge-control/scripts/kg_remap_commits.py \
  --repo "$PWD" --commit-map /path/to/commit-map --report /path/outside/checkout/remap.json \
  --refresh-run-exceptions
```

既定はdry-run。追跡ファイルに既存変更があるcheckoutへの `--write` は拒否する。未解決があると終了値2を返し、`--write` でも書き込まない。短縮形が複数の旧commitに一致する場合・消滅したcommitは自動適用しない。mapにない参照や公開refから到達不能な新commitを報告し、そのまま残すことを確認した場合だけ `--allow-unresolved --write` を使う。SHAだけから別実装への対応を推測しない。第三者repoのURL、パッチ（保存済みの `range-diff.txt` を含む）、削除manifestと内容hashは保持する。

`--refresh-run-exceptions` は既存runの固定hashを先に照合し、参照変更を受けたrunのhashだけを再計算する。例外パスやファイルの集合は増やさず、パッチと削除記録も変更しない。更新後はノード・summaryの差分を確認してskillの完了手順と `kg_repro_paths.py` を実行する。`--text-dir /path/to/exported-markdown` ではissue本文やmodel cardのローカルコピーに同じ更新案を作れる。外部への投稿は別工程で行う。

公開した対応表と旧SHAの説明を保つため、`docs/history-rewrite/` は再実行時の置換対象から除外する。

## Papersの仕様

人間が研究に必要と判断した論文だけを、タスク横断で `Papers/` に採録する。探索しただけでは登録しない。
ディレクトリIDは `paper-<発表年4桁>-<短いkebab-case名>`（年は1000〜9999）。既存IDは維持する。同一文献の版違いは一次資料URLを更新し、別の論文として登録しない。
各ディレクトリには通常ファイル `paper.md` **だけ**を置く。symlink、PDFその他の添付、旧 `pdf` / `sha256` / `license` metadataは拒否する。本文は外部の一次資料URLで閲覧し、永続保存しない。

`paper.md` の必須frontmatter:

| キー | 内容 |
|---|---|
| `id` / `type` | ディレクトリID / `paper` |
| `title` / `year` / `authors` | 正式題名 / 発表年 / 著者名の配列 |
| `tasks` | 関係するタスクの配列（未実験のタスクも可） |
| `source` | 参照した版の一次資料URL（arXivのabsページなど）。[入力値の境界条件](#入力値の境界条件)に従う |
| `summary` | 研究内容の短い要約（非空文字列） |
| `reason` | 採録理由。研究上の問い、または探索・引用元・役割・根拠を含む発見経路（非空文字列） |
| `external_ids` | 正規化した外部IDの辞書。ID未確認の通常登録では空辞書も可 |

本文には追加の読解、プロジェクトとの関係、検証仮説と適用限界を記せる。必須の要約・理由はfrontmatterを正本としてUIが表示し、本文へ複写しない。参照元実験はノードの `papers` から求める。
`authors` / `tasks` は非空・重複なしの文字列配列。title・著者名は空白のみ不可。`tasks` は[タスク登録簿](#レジストリ)のトピック。`source` はHTTP(S) URL、yearはIDの年と一致する整数。
通常登録と探索からの採録の入口は `kg_papers.py`（[登録手順](../.agents/skills/knowledge-control/references/record.md#related-paper)）。探索からの採録は人間の明示操作で、発見経路を `reason` に自動記入する。

## 引用探索文書の仕様

`citation_map/papers/<id>.md` と `citation_map/searches/<id>.md` にYAML frontmatter付きMarkdownを置く。各ディレクトリは通常の `.md` 文書だけを許可し、PDFやsymlinkは拒否する。frontmatterの未知のキー、YAML重複キー・aliasも拒否する。本文は任意の補足で、機械処理はfrontmatterのみを読む。
探索CLI用の型・parse/serialize・入出力・検証APIは `kg_citations.py`、外部IDの共通処理は `kg_identifiers.py`（いずれもknowledge-controlの `scripts/`）。Web UIは同じfrontmatterを読む。

### 入力値の境界条件

`created_at` は文字列で、`YYYY-MM-DDTHH:MM[:SS[.小数秒]]` に `Z`・`±HH:MM`・`±HHMM` のいずれかを付ける。
例えば `'2026-11-14T22:00:00+09:00'` と `'2026-11-14T22:00:00+0900'` はともに受理する。
年は0001〜9999、日付は実在する暦日、時は00〜23、分・秒は00〜59、offsetの時は00〜23・分は00〜59とする。小数秒は秒の後の1〜6桁に限る。
timezoneなし、空白による日付/時刻の区切り、基本形式・週日付、offsetの秒、うるう秒、範囲外の値は拒否する。入力文字列は補正・切り詰め・UTC変換せず保持する。

Papersと引用論文の `source`、および `url` 外部IDは、入力そのものが `http://` または `https://` から始まる絶対URLとする（schemeの大文字も可）。
hostはASCIIのDNS名（各labelは英数字で始まり終わる1〜63文字、中間は英数字・ハイフン、全体253文字以内、末尾のdotは可）、標準の10進4区切りIPv4、またはbracket付きIPv6を使う。IDNはpunycode表記にする。
portを付ける場合は1〜5桁の10進数、0〜65535。credentials、空白・制御文字・BOM、backslash、空host、空port、不正なIP/DNS表記を拒否する。数値・16進数で終わるhostを省略形IPv4へ補正しない。
path/query/fragmentはUnicodeやURLエンコードも保持する。`source` は大文字小文字や明示portを含め入力をそのまま保持し、URL parserによる欠けたslashの補完などはしない。`https:example.org/paper` は拒否する。

受理/拒否と外部IDの入力正規化の共有例は [citation-boundaries.json](../tests/fixtures/knowledge/citation-boundaries.json)。PythonとTypeScriptのunit testはこの1ファイルを読み、同じ期待値を照合する。

### 外部ID・同一視・結合

`external_ids` は次のnamespaceから文字列への辞書。未知のnamespaceは拒否する。登録時に正規化し、保存済みの非正規形は検証エラーにする。
prefix・URL内の固定部分の大文字小文字はASCIIで照合し、Unicodeの類似文字を同一視しない。

| namespace | 正規形と入力の扱い |
|---|---|
| `doi` | `10.<4〜9桁>/<suffix>` を小文字化。`doi:` / `https://doi.org/` / `https://dx.doi.org/` を除去しURLエンコードを復号 |
| `arxiv` | `YYMM.NNNN[N]` または旧 `archive/NNNNNNN`。`arXiv:` / abs・pdf URL / `.pdf` / `vN` を除き小文字化。版は `source` に残す |
| `openalex` | `W<数字>`。OpenAlex URL・大文字小文字の差を正規化 |
| `url` | 他のIDが判らない場合の一次資料URL。上のURL条件を検証し、scheme/hostを小文字化、fragmentを除去、空pathを `/` にし空queryの `?` を省く。path/query・明示portは保持。HTTPとHTTPSを推測で同一視しない |

同じnamespace/valueを1つでも共有する文献を同一視の候補にする。候補が1件でも、両側にある全namespaceの正規化した値を照合し、1つでも異なれば結合・別名の集約をエラーにする。例えばarXivが一致してもDOIが異なる場合は拒否し、誤ったIDを明示的に訂正してから再検証する。片側にしかないnamespaceは矛盾とせず、別名として追加できる。この規則はPapers↔探索の結合と、探索文書どうしの同一視に共通で、役割や題名で矛盾を推測修正しない。

複数IDは1文献の別名としてまとめ、別名を追加しても既存の `identity_key` / 文書IDを変えない。新規の `identity_key` は `doi` → `arxiv` → `openalex` → `url` の優先順で `namespace:value` を選び、文書IDは `cite-` + そのUTF-8文字列のSHA-256先頭16桁とする。外部IDは最低1つ必要。別名追加が既存の複数文書を結ぶ場合は暗黙に片方を選ばずエラーとする。1文書に別名を集約し、引用・探索の参照をそのIDへ置換してから再検証する。題名の類似や発表年だけで同一視しない。

Papersとの対応も外部ID集合の共通要素から導出する。探索文書にPapersのIDを、Papersに引用文書IDや手書きの逆リンクを持たせない。各領域内の外部ID重複と、複数のPapersを1引用文献へ結ぶ曖昧な結合はエラー。外部IDなしのPapersは結合対象外。Papersの書誌・要約は採録時点の登録情報で、探索側の引用判断はPapersへ複写しない（採録理由の発見経路だけは登録時点の来歴として記録する）。

### 論文文書 (`type: citation_paper`, `schema_version: 1`)

| キー | 内容 |
|---|---|
| `id` / `identity_key` | 上記の不変ID / 外部ID集合に含まれる同一視キー |
| `title` / `year` / `authors` | 題名 / 年（不明はnull） / 著者配列（不明は空配列）。titleは必須 |
| `external_ids` / `source` | 外部ID辞書 / 一次資料HTTP(S) URL |
| `summary` | `available` では非空の研究要約。未読・取得失敗では空文字列を許可し、内容を捏造しない |
| `text_status` | `available`（取得済み） / `failed`（本文取得失敗） / `not_requested`（未展開の参照先） |
| `text_reason` | 取得状態の補足。`failed` のとき非空必須。他は空文字列も可 |
| `references` | 下記の参考文献行の配列。取得失敗・未読では空配列 |

参考文献は本文内の言及回数ではなく参考文献表の1行ごとに記録する。

| 行のキー | 内容 |
|---|---|
| `ref_id` / `citation` | 引用元内で一意な `ref-<数字>` / 原文の書誌表記（非空） |
| `target_id` | 同定済み文献の文書ID、未同定はnull。引用先の正規書誌・外部IDはこの文書から結合して得る。行に複写しない |
| `identification_status` | `resolved` / `unresolved`（参照先未同定）。target_idの有無と一致させる |
| `roles` | 下記の役割の配列。重複不可、複数可 |
| `decision_status` | `classified` / `unclassifiable`（役割判定不能） / `pending`（未判定） |
| `reason` / `evidence` | 判断理由（常に非空） / 根拠位置の文字列配列（節・表・ページ・本文引用等）。`classified` は1役割以上・1根拠以上必須。他の状態はrolesを空にする |

未同定でも引用文脈から役割を判定できるため、同定状態と判定状態は独立させる。同じtargetへの複数行は許可するが、集計では引用元・引用先・役割を一組として数える。

| rolesの列挙値 | 親 #2011 の5分類 |
|---|---|
| `foundation` | 基盤アーキテクチャ・モデル |
| `component` | 個別技術・構成要素 |
| `comparison` | 比較対象 |
| `related` | 直接関連する研究・機能 |
| `benchmark` | ベンチマーク・データセット |

### 探索文書 (`type: citation_search`, `schema_version: 1`)

| キー | 内容 |
|---|---|
| `id` | `search-<小文字英数字のkebab slug>`。実行ごとに一意。呼出側が時刻や実行名を含めて割り当てる |
| `created_at` | [入力値の境界条件](#入力値の境界条件)に従うtimezone付きISO 8601日時の文字列 |
| `root_id` | 起点論文の文書ID。入力URL/IDは起点文書のsource/external_idsから解決 |
| `roles` / `max_depth` | 展開に使う役割（非空、複数可、重複不可） / 0以上の整数。起点は深さ0 |
| `generations` | `depth`、`paper_ids`（到達した文献ID）、`expanded_ids`（その世代で本文取得・判定を試みたID）の辞書配列 |
| `issues` | `paper_id` と `ref_id`（文献全体の問題はnull）からなる問題箇所への参照配列 |

世代は0から連続し、指定役割の `classified` かつ `resolved` な辺だけをたどる。既出文献は再度世代へ入れないが、論文文書の全ての辺は残す。各 `paper_ids` は重複不可。最大深さでは `expanded_ids: []`、それより浅い全到達文献は取得を試み、`not_requested` のままにしない。次世代の集合は前世代のexpanded_idsから得られる未出の引用先と一致させる。文献が尽きた時点で止め、空世代は作らない。最大深さを超える世代・参照切れ・不足/余分な展開はエラー。

`issues` はexpanded_ids内の `failed` 文献と、`unresolved` または `unclassifiable` / `pending` な行を漏れなく指す。状態・理由・根拠は論文文書から求め、探索側へ複写しない。これにより探索ごとの取得失敗・未同定・判定不能を読める。未展開の末端文献は取得失敗に数えない。保存済み文献を再利用し、同じ引用元の再読・再実行で辺の数を増やさない。論文文書を訂正した場合は、影響する探索文書の世代・issuesも同じ変更で更新する。

Papersへの採録対象は起点と、その探索の各 `expanded_ids` が記録した **全ての `resolved` 引用先**。指定役割・判定状態・`generations.paper_ids` への収録は採録の条件にしない。比較探索でも、基盤として表示された未展開の引用先を人間が採録できる。発見経路には、探索内で展開した全ての引用元についてref_id・役割・判定状態・判断理由・根拠を記入する。判定不能や根拠の空欄もその状態を明示する。未展開論文のcached参考文献にしかない引用先や、探索範囲外の文献は拒否する。

固定例は `tests/fixtures/knowledge/citation_map/`。A→B→D（比較）、A→C（基盤）、B/C→Dの合流、A→Bの比較・基盤兼用、本文取得失敗E、未同定行と役割判定不能行を含む。テスト用の架空文献で、本番のPapersや探索一覧には入れない。

## summaryと履歴

`summary.md` は自動生成したノード一覧ではなく、タスクごとの**現在の判断**（採用中の構成と根拠、未解決課題、次の実験）だけを書く入口で、20,000文字以内に保つ。判断が変わったら該当節を書き換え、日付つきの記録を追記しない。確定した経緯はgroupノードにまとめる。2026-11-05までの日付順の本文は `history/summary-2026-11-05.md` に凍結した（追記しない）。更新手順はSKILLに集約する。

`--check-summary` は、本文の有無と文字数、summary・README・reportsの相対リンク、historyからライブラリ内へのリンクを検査する。`--base-ref` と併用すると、追加したノードと `status` を変えたノードに `summary_impact` を要求し、`updated` を宣言したのにsummary.mdが変わっていない場合と、`deployments.yaml` が変わったのにsummary.mdが変わっていない場合をエラーにする。この検査は照合したという宣言を求めるだけで、判断の正しさは保証しない。

## 移行・検証

既存のflat形式は `kg_migrate.py`（dry-run、適用は `--write`）で移行した。不明な分類は `--task-map <JSON>` で明示する。混在状態はエラー、移行済みの再実行は無変更。保守対象Markdownの相対リンクを移動先に合わせて更新する。初回移行後もID、metrics、config、graph関係、再現bundleの内容を保持する。

```bash
.venv/bin/python .agents/skills/knowledge-control/scripts/kg_validate.py --check-summary
```

`--base-ref` を付けない検証は、ブランチの変更に対する検査（番号・identity、削除記録、追加ノード、summary照合）を実行せず、その旨を表示する。

run bundleの`repro.sh`は、checkoutするcommitで追跡されたファイル、bundleに保存したファイル（`$SCRIPT_DIR/...`）、
保存した`uncommitted.patch`が追加するファイルだけをscriptとして参照する（`*.py`/`*.sh`/`*.yaml`、`-m` module、`PYTHONPATH`）。
検査器が認識する外部CLI moduleは、そのcheckout commitの`uv.lock`に固定されている場合に限り利用できる。
CLIへ渡すtest等のscript参照は、引き続き上記の保存先で検証する。
`/tmp`や`outputs/`のscript、repo外でbuildした拡張は、bundleへ保存するかrepro.sh内でbuildする。

```bash
.venv/bin/python .agents/skills/knowledge-control/scripts/kg_repro_paths.py
```

PRの番号衝突・既存identityの変更は、マージ先commitを比較元として追加検証する。既存IDのtask / sequence / recorded_at変更、counterの削除・縮小、新規IDによる予約済み番号の再利用をエラーにする。追加したノードについては、廃止したトピックへの登録とartifactsのホームディレクトリ絶対パスをエラー、issue未記録のrunをWARNにする。比較元は、ライブラリを含むGit checkoutの履歴から読む。比較元が初回移行前のflat形式なら番号履歴の比較は適用しない。参照先はローカルにfetchしておく。

```bash
.venv/bin/python .agents/skills/knowledge-control/scripts/kg_validate.py --check-summary --base-ref origin/main
```

CI `Knowledge library` は関連ファイル（標準pipelineのconfigと `LICENSES.md` を含む）のPR・main push・手動実行で動く。上記検証、CPUのみのstorage/登録/曲線テスト、SKILLの例を実行して成果物を検証するテスト、Web UIのlint/test/build/E2Eを行う。PRはbase commit、main pushは直前commitと比較する。手動実行は現在の構造を検証する。

ERRORは修正必須。WARNは追加したrunのissue未記録だけで、任意の追跡情報の不足としてCI失敗にしない。科学的な妥当性・実験間の公平性・本文と数値の意味的一致は機械検証の範囲外。

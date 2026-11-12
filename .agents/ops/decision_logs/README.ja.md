# 親の判断記録

[English](README.md) · [简体中文](README.zh-CN.md)

結果・親の判断・行動・訂正を表す共通契約version 1、非公開snapshotの保存、
公開用への変換と既存parent・runner・review操作の捕捉を提供する。
親ごとのGit公開と階層READMEの決定的生成も提供する。記録は実行権限や成果受入の判断を与えない。

## 契約と責務

フィールド・列挙値・検証の正本はコードに置く。既存の
[workerログ](../worker_logs/README.md)の3ファイル形式は変えず、schemaの部品、
`Guard`、重複キーを拒否するJSON decoder、flock、永続化するatomic writerを直接再利用する。

| モジュール | 利用者との契約 |
|---|---|
| [schema.py](schema.py) | `new_id()`・`validate_record()`・`json_bytes()`。厳密なversion付きフィールドと欠測表現 |
| [graph.py](graph.py) | `validate_graph()`・`references()`・`states()`。型付き参照、DAG、行動履歴、明示的な検証 |
| [journal.py](journal.py) | `Journal`。原文の固定と準備・公開の追記履歴 |
| [projection.py](projection.py) | `render()`・`validate_public()`。公開単位 `{record, sanitization}` |
| [aggregation.py](aggregation.py) | `aggregate()`。すべての原記録を参照する表示用の集約 |
| [navigation.py](navigation.py) | `generate()`・`validate_links()`。階層の決定的生成とリンク検査 |
| [presentation.py](presentation.py) | 文章・旧summaryの構造値・変更点・補助観測を、安全で上限付きの表示項目へ変換 |
| [publication_policy.py](publication_policy.py) | 独立した追加専用例外、再生成バイト列・PR同一性の検査 |
| [publication.py](publication.py)・[publication_state.py](publication_state.py) | 明示公開、非公開batch、応答喪失からの照合 |
| [capture.py](capture.py) | 非公開outbox、行動snapshot、状態表示、捕捉だけの再送 |
| [parent_commands.py](parent_commands.py)、[parent_sources.py](parent_sources.py) | 判断入口、CLI結果、正本コメントと明示検証 |
| [observations.py](observations.py)、[review_capture.py](review_capture.py) | runner・dispatcher・workerログ・取得・review・mergeの既存の事実 |

各記録は独立したUUID4 hexのID、種類、所属、区別した3つの時刻、型付きdataを持つ。
所属にはrepo・親・nullableな担当issue・活動の公開キー・既存workerの公開委任キー/turn・
nullableなreview公開キーを持つ。親の直接作業に架空の子issueは不要。
新規ID・キーは `new_id()` を使い、私的session/run IDや配送キーから導出しない。
対応はcapture entryの非公開contextだけに残す。

`validate_record()` は入れ子を含めキー・型を厳密に検査し、JSON入力では共通decoderで
重複キーを拒否する。`validate_graph()` は欠落・種類違い・別の親への参照、循環、
行動履歴の衝突、判断planとの不一致も拒否する。時刻の近さで因果を推定せず、
1結果から複数の判断・行動への分岐と、複数結果から1判断への合流を扱う。

## 結果・判断・実行状態

結果は機械観測・worker報告・親の検証を区別する。runtime終了、成果の品質、CI、
finalの有無、公開は別の観測。ファイルの存在で親の受領・承認を推定しない。
親の明示的検証は、検証対象を `verifies` で参照する結果として追加する。
`states()` はこの参照から未判断・確認未記録を導く。

結果は `previous_result_id` に対するコード・要件・具体指示・実行条件・確認済み範囲の
差を持つ。比較対象がなければ全項目を `unconfirmed` にする。
根拠には固定revisionか内容fingerprintを持たせる。編集可能なURLとfingerprintだけで
全文復元を保証しない。非公開・取得不能の根拠は私的URLを出さず、理由を残す。

判断は入力結果ID、短い理由、具体的な次の行動plan、再利用/再確認範囲、未解決事項、
再開条件と、設定・人間指示・親の選択を区別する出典を持つ。
初回判断は `basis` に要求・合意の固定参照を置き、先行結果を捏造しない。
planにも独立した公開IDを付け、分岐ごとに別のplan・行動を作る。

行動は判断とplanを参照する。判断IDが渡されなかった場合は両IDをnullにし、
`decision_missing_reason` を必須にして判断を作らない。
状態の更新は `previous_action_id` を持つ新たな行動snapshotで表し、所属・判断・plan・
操作種類を保持する。結果は先行する要求済み/実行中の行動snapshotを参照し、
その後の終了snapshotが `result_ids` で結果を参照する。同じsnapshotを双方から
参照すると循環になる。`decided_not_executed` の行動から結果は作れない。
行動履歴は `unknown` を挟んでも確認済みの実行事実を保持する。
実行が観測された同じ行動を、要求済みや起動前拒否へ戻すことはできない。

訂正は対象と理由を持つ追記。必要なら同じ種類・所属の別の置換記録を参照する。
元記録は残り、graphは暗黙に置換したり、訂正を新たな観測へ読み替えたりしない。

## 欠測・時刻・使用量

取得状況は `{status: observed, value: ...}` または
`{status: unavailable, reason: ...}`。不明な時刻・時間・使用量・実効設定を、
0・選択モデル・現在時刻で補わない。観測した0は有効。
発生/受信時刻は取得状況を持ち、`recorded_at` は実際の記録時刻。
観測した時刻はすべてtimezoneを必須にする。

選択model/effortは指定元を保持し、実効値は独立して観測する。計測値は根拠を持つ。
使用量はturn単位/累積の区別と公開counterキーを持つ。
集約・本パッケージは累積値を加算せず、金額を推定せず、cache分を課金量へ読み替えない。

最終報告欠測は `missing_final` などの理由を残す。未判断、親の確認未記録、
判断済み未実行、起動前拒否、公開待ちは別のフィールド/状態として保持し、
runtimeの終了値から推定しない。

## 捕捉元と未対応入口

結果のevidenceに `capture_source`・artifact・観測snapshotのfingerprintを残す。
非公開根拠は公開不可と明示し、fingerprintだけで全文を保存したとは扱わない。
親の証拠用Issue（旧親コメントを含む）、公開済みreview・workerログは取得できた固定の公開根拠も参照する。
outboxは非公開の配送入力であり、公開単位ではない。
親の応答の要約は公開可能な項目だけに絞り、assignmentや任意の入れ子の状態を丸ごとコピーしない。
私的な来歴IDは捕捉前に登録する。完全な応答とエラー詳細は非公開根拠に残す。

| 入口 | 取得元と境界 |
|---|---|
| `run-worker` | parent CLIの要求・拒否・応答、結び付けたrunnerの `result.json`、選択/実効設定・取得できた使用量・終了理由・finalの有無 |
| `start-worker` / `continue-worker` | CLIの要求・拒否・予約、dispatcherのjob対応・永続化した取消/runner終了、runnerの保存。予約を起動と推定しない |
| dispatcherからの直接の子起動 | bootstrapの子を含む同じjob対応・runner callback。親の判断を捏造しない |
| `record-worker-run` / `record-worker-ended` | 既存runner根拠の明示登録 / 親による終了確認 |
| `record-result` | 保存された検証とPR・証拠。明示検証が捕捉済み結果を参照する。`review-request --prs` 内も同じメソッドを使う |
| `note` / `decide-worker-plan` | 正本コメントへの参照 / 保存済み計画回答の結果。共通判断を自動で作らず、`decide` から理由の正本を参照できる |
| `improvement` / `improvement-publish` | 候補観測・明示的な切り出し判断を正本コメント参照で記録し、起票要求/結果へ接続。元親の担当中の操作だけを捕捉する |
| `agree` / `basis-record` / `create-child` / `workflow-report` | 根拠記録・子作成・全体受入の既存結果。改善親のagreeは人間承認を観測した結果として出典コメントを参照し、実行済みと扱わない |
| `delegate`、`stop` / `resume` / `stop-worker`、`handoff` / `release` | CLI結果と正本コメント。dispatcherの取消とrunnerの停止/復旧は別の事実 |
| `review-prepare` / `review-request` / `review-state` | 固定要求、明示指示・状態の結果。親の受領/検証は明示操作でだけ記録する |
| review `run` | 既存execution・output・result・公開状態。runtime失敗時の有効な出力、exceptionと先行reviewの公開キーも保持。公開で元resultを変えず、消費済みrunの取得は新しいAIレビューにしない |
| review `recover` / `resume-publication` / `complete-evidence` / `resolve` | 既存の保存済み根拠の復旧、公開だけの再試行、親の明示検証。前回結果を保持し、新しいAI起動にしない |
| reviewの取得tool | CI取得失敗を含む既存の取得receipt。同じreview対応・tool・引数の再試行を `retry_fetch` にする。変更なしと分かるのは取得指示であり、通信条件等を補完しない |
| merge gate `merge` | PR別の判定、保存済みmerge試行/結果、全体結果。dry-runの合格と実mergeを分ける |
| workerログの受信/公開、`worker-log-status` / `publish-worker-log` | 既存journalの状態、final/公開失敗、main保存確認。公開だけの再試行は `retry_publication` とし、前回結果を参照する |

対応する要求には `--decision-id` と任意の `--plan-id` を渡せる。一致planが1件なら選び、
あいまいな指定・操作種類違い・要求済みplanは拒否する。判断なしでは両IDをnullにし、
理由未記録を明示する。`--retry-of` は前回の結果ID、`--changes-file` は契約の比較5項目を受け取る。
同じ行動のsnapshotは `previous_action_id`、別試行の比較は `previous_result_id` で結ぶ。
同じAIの継続は正確な前回runtime対応を参照する。継続要求が初回起動へ解決された場合は、
要求と別の `start_ai` を残す。既定モデルをrunner引数へ展開する呼出元でも、設定由来と明示選択を区別する。
保存済みpreflightから要件・送信指示のsnapshotを比較し、開始revisionの変化も検出する。
commitが同じだけで作業中コードも同じとは認定せず、未観測の条件・確認済み範囲は未確認のまま残す。

未対応は、任意のshell/HTTP/gh操作、このCLI文脈外のParentSession直接呼出し、
親対応のないrunner、親namespaceが定まる前の引数parse・identity・repo解決失敗。
過去の会話や任意の過去runは復元しない。reviewの修正確認・復旧・モデル選択方針を変えず、
捕捉のための定期poll・AI再起動・Git公開は追加しない。

## 親の判断を明示記録する

同じparent stateを使い、担当sessionから実行する。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo OWNER/REPO --parent N decision-log-status
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo OWNER/REPO --parent N decide --spec tmp/decision.json
```

specは `inputs`（複数可の結果ID）・`evaluation`・`reason` と、任意の
`next_actions`・`reuse_scope`・`recheck_scope`・`unresolved`・`resume_conditions`・
`basis`・`source_kind` を使う。planはschemaのoperation/summary/actor/target/instruction/
conditions/expected_resultを指定し、IDはCLIが割り当てる。初回判断は `basis` に固定した公開の要求/合意根拠が必要。
同じsessionで同じspec・根拠snapshotを再送すると、同じ判断/plan IDを返す。
要求されていないplanは、判断しただけで未実行のまま残る。

`reason` の代わりに `reason_ref` を1つ指定できる。

| kind | 既存の正本への参照 |
|---|---|
| `note` | noteコマンドが返した `body_sha256` |
| `worker_plan` | `child` と計画回答コマンドの `decision_id` |
| `model_reason` | `child`。現在保存されているモデル選択理由 |
| `review_reason` | `run_id`。固定review要求の理由 |
| `improvement` | `key` と候補の `event_id`。元親の正本コメントにある切り出し判断 |

`note` と `improvement` は[親の共通reader](../parent/README.md#機械証拠の保存共通reader)から検証し、実際の媒体のcomment URLを参照する。
理由本文を公開記録へ再コピーせず、参照/fingerprintを保持する。
親のactorキーは私的sessionに対応するランダムな公開キーで、引継ぎで過去の判断者を変えない。
次の対応CLI（review/mergeも含む）へ返された判断IDを渡すと、後続結果が行動を介してつながる。

終了時改善の `improvement-decide` は候補正本への人間判断を記録するが、終了済み元親のjournal公開権限を広げない。
改善親側のagreeで承認を観測し、別graphの記録IDを直接つなぐ代わりに元コメントURLを参照する。
手順・人間確認の保証範囲は [終了時改善](../parent/improvements.md)を参照する。

## 捕捉失敗と復旧

journal配送より先に、親journal横の非公開 `captures/` outboxへID・時刻・取得snapshot・
配送入力を固定する。flockで並行更新を守り、呼出元とrunnerは最新outboxを読み直す。
障害はstderrへ警告し、配送待ちを保持する。別の非公開診断パスで、outbox書込み失敗や
結果を構築する前の変換失敗も検出する。state filesystem全体が壊れた場合はstderrでのみ通知でき、
壊れた媒体への永続記録は保証できない。
捕捉の初期化では明示入力の不正とjournal/outbox障害を区別する。媒体障害では判断IDが
指定されていても元の操作を維持し、永続的な未捕捉診断を残す。結果を構築する前に
失った事実は、配送だけの再送では欠測を解消したことにしない。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo OWNER/REPO --parent N decision-log-status
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo OWNER/REPO --parent N retry-decision-capture
```

statusは取得元・行動/結果ID・配送待ち/未終了receipt・未捕捉診断・公開状態と、無害化した読み取り表示を返す。
表示のblockedは公開成功ではない。再送は固定記録を依存順に配送するだけで、worker・review・取得・mergeを再実行しない。
読めないreceiptや失った取得snapshotを空のjournalへ置換せず、未完了として残す。
変換/取得元の欠測は、修復後に既存の正本根拠を再登録する必要がある場合がある。
観測の失敗で元の操作の終了値・状態を変えない。`decide` と捕捉復旧コマンド自身の記録/復旧が失敗した場合は非0を返す。

## 非公開journalと公開境界

既存の[parent state](../parent/README.md)を使い、
`Journal(parent_store.directory, repo, parent)` を作る。
`decision-logs/` 配下にrepo・親を分離した親ごとのjournalを置く。
ディレクトリはprivate、ファイル/lockは所有者のみアクセス可とし、symlinkを拒否する。
entry配列は追加だけ。flock・ファイル/ディレクトリfsync・atomic replaceで追記する。
破損JSON、重複/未知キー、未知version、所属の変化を空状態に置き換えない。
保証範囲はローカルfilesystemで、分散した担当排他ではない。

- `capture(key, record, private=...)` は入力を固定する。非公開contextはsession、run、
  その他のID、checkout root、nullableなcheckout commitを持つ。
  記録IDを省略するとlock内でUUID4を割り当て、同じprivate配送キーの再送では
  同じIDを返して書き込まない。入力変更は拒否し、呼出側は時刻・所属も固定版を再利用する。
- `prepare(record_id, key=..., at=...)` は安全な公開単位を固定するか、原文を残して
  `blocked` を追記する。依存先から準備する。blockedの再検査は `retry=True` と
  新しい配送キーで行い、元の作業は再実行しない。
  準備時の非公開な変換条件と内容の照合値も固定する。journalの読取はこのsnapshotを
  検証し、後から判明した私的IDで再renderしない。readyになった変換は置換できない。
- `publication(record_id, key=..., status=..., at=..., evidence=..., error=...)`
  は捕捉と独立してpending・blocked・publishedを追記する。
  publishedは固定した公開根拠を必須にし、終端状態とする。
  Git/HTTPの実際の保存確認は呼出側の責任で、本libraryの認定ではない。
- `public_payloads()` は依存順の検査済み公開単位だけを無害化件数付きで返す。
  公開側はこの単位を使って再検査する。`public_records()` はgraph/集約用の記録だけの表示。
  `status()` は非公開対応を出さず、捕捉・公開・欠測の状態を返す。

公開用の取得とpending/publishedの公開記録では、固定したenvelopeを現在判明している
私的IDで検査する。新たに判明した私的値で公開が拒否されても、そのcaptureは保存される。
呼出側は `publication(..., status="blocked", error=...)` で公開失敗を記録し、
元のenvelopeを保持する。共有workerログguardの従来の置換順序は変更しない。

共通のworkerログguardで文章を無害化し、全公開fieldと無害化metadataを再検査する。
既知の私的IDとlocal pathを件数付きでマスクする。
固定参照URL・ID・モデルmetadataは意味を書き換えず検査する。
残った資格情報、未知UUID/session表記、危険なURL、制御文字は準備を拒否する。
未知の秘密を完全に分類する機構ではない。Git path・branch・生成README・PR本文は
公開側でも公開検査する必要がある。

## 表示用の集約

`aggregate(records)` は、対象・所属・状態・行動・条件キー・モデル設定が同じで、
比較参照が連鎖し、全変更が明示的に変更なしである隣接した機械確認/取得再試行を集約する。
捕捉側は同じ条件にだけ同じランダムな条件キーを使う。
未確認/変更あり、AI起動/継続、判断、行動snapshot、訂正は境界として残す。
1つの取得行動内の試行は集約でき、別の行動は表示を残す。
後の判断・検証・訂正・行動の結果参照・比較分岐が参照する途中の観測も、
参照が系列の後に現れる場合を含めて表示を残す。集約できる参照は次の標本への単純な比較鎖だけ。
全記録ID・件数・初回/最終記録時刻・最終記録IDを返し、利用側は最終記録から結果を引く。
原記録と途中の失敗は残り、時間や使用量は補わない。

## Git公開と閲覧

親の判断を記録した後や、まとまった結果を受領した区切りで、担当親が明示実行する。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo OWNER/REPO --parent N publish-decision-log --dry-run
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo OWNER/REPO --parent N publish-decision-log
# blockedの原因を照合・修復した後:
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo OWNER/REPO --parent N publish-decision-log --retry
```

timer・定期poll・捕捉ごとの自動公開は追加しない。既存parent stateと担当sessionを照合し、
issueのclose後の履歴公開も同じ本人確認を行う。新しい実装権限は与えない。
`Journal.prepare()` で未準備の捕捉を変換し、`Journal.public_payloads()` のうちreadyかつ
未公開の全記録を親ごとの1 batchへ固定する。blockedは状態に残し、独立して安全な記録は公開できる。
未完了batchを照合してから次を作る。batchの途中で増えた捕捉は、そのbatchのmerge後の
次の明示呼出しでまとめる。

既存workerファイルを維持した公開構成:

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
          # 既存instruction.md・final.md・meta.jsonは変更しない。
```

eventの正本は共通の `{record, sanitization}` envelopeそのもの。READMEは、そのHEADに
ある公開記録と既存workerファイルから生成する。非公開の捕捉・公開stateやLLMを参照しない。
因果参照に沿って並べ、独立した記録間だけ時刻・IDで順を固定する。分岐・合流を含む活動を分け、
**時刻・対象｜結果｜親の判断と理由｜次の行動・実行状態｜根拠** の表にする。
結果・判断・行動snapshot・訂正はそれぞれの参照と逆参照を残す。原文はescapeし、
MarkdownリンクやHTMLとして挿入しない。各階層に上下・活動へのリンク、反映した全記録への導線、
記録時刻の範囲を置く。未判断、判断済み未実行、最終報告などの記録不足を明示する。
補助情報は折りたたみに置き、時刻・設定・使用量の欠測と累積値を保持して加算しない。
`turn-<sequence>/README.md` は安定した閲覧先で、連番からinitial/resumeを推測しない。
workerファイルが後着したら既存フォルダーのREADMEと相互にリンクし、起動前拒否後の
initial再試行でも先に公開したターンのアドレスを保つ。指示・最終報告の本文は複製しない。

新しい記録を持つ親について、既存ターンに指示・最終報告への導線を生成し、対応記録がなければ
**導入前・判断未記録** を表示する。新形式の記録がない親は既存ディレクトリへのリンクを維持し、
公開snapshotから区別できない導入時期・公開待ち・捕捉不足は未確認と明記する。
過去全ログを一括移行しない。失敗ターンには、公開finalがあると装わずREADMEを置ける。
公開snapshotだけでは非公開の待機件数や自身のmain反映を判定できないため、各ページの
**公開待ち** 表示は `decision-log-status` での照合へ案内する。merge後の公開事実だけを理由に
READMEを再更新しない。

### 概要の配置とページ上限

[親一覧](../../logs/parents/README.md)から親の概要へ進む。親・子・委任・ターンのREADMEは、
件数・欠測状態・最新の観測結果/行動と、直近最大6件の判断を示し、全記録を複製しない。
判断には入力結果・理由・再利用/再確認範囲と、実際の後続結果を併記する。
活動名は記録された操作/コマンドと対象から作り、記録期間を別に示す。
不透明な活動キーや時刻の近さから、名前・判断・受入・因果関係を補わない。
最終報告の欠測件数では、取得元が明示した `not a final-report observation` を観測対象外として分ける。
それ以外の欠測理由は残す。未解決事項は、その判断を記録した時点の記述として扱う。

詳細の5列表と補助情報は活動ページへ置く。12記録までは活動READMEに収め、超える場合は兄弟の
`activity-<public-key>-records-<sequence>/README.md` へ分割し、前/次ページをつなぐ。
判断のない機械観測を含め、全記録に正規のアンカーを1つずつ持ち、上位から同じアンカーへリンクする。
同条件の観測は既存の損失のない集約件数を表示できるが、判断を作らない。

一覧は最大6項目を概要に置き、全一覧を40項目ずつ分割する。生成先は既存のREADMEパターン内とし、
rootの `parent-index-*`、親配下の `activity-index-*`、子配下の `d-index-*`、
委任配下の `page-*`、ターンフォルダーへのsuffixを使う。これらは閲覧ページであり、
架空の親・活動・委任・ターンではない。eventは移動しない。
固定した項目数でページを割り当て、記録やworkerファイルの追加でも既存ページのアドレスを保つ。
worker一覧は後で判断記録が加わった親も含めた全workerログの親を載せ、新形式導入で一覧を縮めない。
既存READMEの本文は生成の入力に使わない。

各ページのUTF-8全体を192 KiB未満とし、超過した生成は拒否する。詳細1記録の表示予算は10 KiBで、
長い文章は省略を明示し、完全な不変記録へリンクする。原記録や固定envelopeは切り詰めず変更しない。
v1のsummary内にあるdict/listのliteralは、サイズ制限のあるliteral解析で項目表示へ変換する。
コードは実行しない。解析できない構造は原記録の確認が必要と明示し、Python/JSONのまま表示しない。
解析した値を含め、すべての文章をescapeする。
復号した構造・HTML entityを戻した文章・表示用に整形した値は、HTML escapeや省略の前に共通の
public `Guard` を通す。切詰め後の文章と区切り文字を付けた項目名も再検査し、違反は固定の公開エラーコードで
生成を拒否する。問題の値をマスクして表示を続けない。

## 限定公開例外と復旧

[既存workerの3ファイル例外](../worker_logs/README.md#専用prと限定merge例外)とは別の条件とする。
専用branchは `decision-log/parent-<P>/<random-batch-key>`。公開用のdetached worktreeは
元repoの `.claude/worktrees/decision-log-<batch-key>-<revision>` にだけ作る。
各revisionは固定mainを唯一の親とする1 commit。
以下の機械検査を満たしたこの専用PRだけが、Codexレビュー・`Python`・`AGENT_AUTO_MERGE` を
要求せずcreate/mergeできる。

- 全raw差分、旧/新mode、blobのバイト列を確認する。新規eventは親/活動/IDの正規パスにあり、
  固定したjournal envelopeと一致すること。既存eventとworkerファイルは不変。
  削除・rename・symlink・実行可能ファイル・gitlink・余分なファイル・コード/規則の変更を拒否する。
- schema・graph・metadata・本文・パス・リンクを検査する。必要なREADMEがすべて存在し、
  PRのHEADの正本からの再生成とバイト単位で一致すること。任意の文章、欠落ページ、
  正本から生成されないREADMEの変更を拒否する。
- origin/repo、branch、固定marker、HEAD、mainというbase ref、件数と、全ページの
  GitHub PRファイル名/変更種別を確認し、merge直前にPRを再取得する。

merge前に最新mainをfetchし、全記録の和集合から再生成する。mainが進んだら新しい検査済み
revisionを作り、正確なleaseで同じPR branchを更新する。他の親・子の記録とリンクを保持する。
3回mainの前進を観測したら、待機やpollをせず `pending / main_advanced` で後の明示再実行へ返す。
GitHubの古い `base.sha` はsnapshotであり、main更新を消す理由にしない。
HEAD指定でmergeし、実際のmerge treeの生成内容、mainの祖先関係、保存したenvelopeを照合する。
最終fetchからmergeの間の外部更新もmerge treeで検出する。生成が不一致なら
`readme_refresh_required` で止め、成功とはしない。明示的な `--retry` はmerge済みPRを
非公開batch履歴に残し、同じ固定envelopeの `-repair-<number>` branchを作る。
記録がmainに既存なら、この修復PRは正本から再生成したREADMEだけを変更できる。
修復を装うeventの追加・改変は行わない。生成内容が保存済みなら後の呼出しでPRは増えない。
batchのmerge後にworkerファイルが届いた場合も、未公開captureがなくても同じ明示修復で
閲覧内容を更新できる。修復のための捕捉記録は作らない。
全記録の公開確認が済んだ後のrenderer更新も同じ経路で扱う。保存済みmerge事実とbatch全記録の
published事実で過去の確認を識別し、PR/HEAD/mergeの同一性・Git構造・祖先関係・現在のenvelopeは再照合する。
旧表示を新rendererとの一致で認定し直さず、mainの古い表示は明示的な `--retry` で更新する。
修復revisionには現行policyの全検査を適用する。未確認の旧batchはrenderer変更を理由に免除しない。

捕捉journalの隣の非公開 `publication.json` にbatch、revision、送信試行、診断を残し、
flock・fsync・atomic writeを使う。`Journal.publication()` は記録ごとの
pending/blocked/publishedの事実だけを追記する。公開コマンド自体を捕捉へ送らず、
公開が次の公開記録を生むループを避ける。commit直後の中断は完全照合したcommitを再利用する。
push・PR作成・mergeの応答喪失は同じbranch/marker/HEADを照合し、不明なPOST/mergeを
盲目的に再送しない。open PRが既知の古いrevisionを示す場合は、mainがさらに進んでいても
最新revisionのローカルcommitとlease付きpushを復旧してから最新HEADを照合する。
PRへ未反映のrevisionも含め、すべてのrevisionを非公開履歴に保持する。
merge確認後の記録別journal追記が中断しても、次のbatchを選ぶ前に照合する。
dirty worktreeや破損journalを自動resetせず、影響する公開だけを確認・修復する。
確認後はcleanな公開worktreeを削除し、dirtyまたは同一性未確認のものは保持して
`retained_worktrees` の件数で返す。

`--dry-run` は `Journal.preview_publication()` とローカル `origin/main` を使う。
readyなenvelopeは固定版を保持し、未準備の捕捉は同じrenderer/guardで依存関係も含めて読み取り専用変換する。
`prepared_records`、`preview_records`、`preview_blocked`（IDとエラーコード）、
`generated_readmes`、`preview_readmes`（パスとMarkdown本文）を返し、初回も内容を確認できる。
プレビューで拒否した記録はblocked件数に含む。`--retry --dry-run` はblockedの再検査を準備の固定なしで試す。
不可逆な準備、fetch、journal/ref更新、lock作成、worktree作成、GitHub呼出しは行わない。
プレビューはローカルのsnapshotであり、後続の捕捉・guard条件・main更新で実際の公開は変わり得る。
`saved_to_main` は現在の全捕捉の公開照合が済んだときだけtrue。
batchがmergedでも、新しい捕捉やblockedが残れば未公開となる。exit 0のpendingを成功と扱わず、
errorかblockedがあればexit 2。

workflowの正確な除外パターンは `ci.yml`・`pull-request-labeler.yml`、追跡許可は
`.gitignore` を正本とする。既存3ファイル、新event、各階層READMEだけを扱い、
公開実装や任意のrawログを除外しない。`paths-ignore` はパスだけを見て、変更種別や内容は
検査しない。既存eventの編集、READMEへの手書き追加、新形式とworker 3ファイルの混在だけの
通常PRでもCIとlabelerは省略され得るが、限定公開例外を得るわけではない。
専用公開コマンドが全内容とREADME再生成のバイト単位一致を検査し、通常の `merge_gate.py` は
レビューと対象HEADの `Python` 成功を要求して、check欠落を拒否する。
この2つの検査が境界を保証する。workflowの省略自体は内容の保証ではない。
GitHub native path filterの制約は残る。実workflowの省略とGitHub表示は実装merge後の #1817 で確認する。

## 検証

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

分岐/合流、明示した欠測、負例、無害化、集約、実際の2process追記、fsync中断、
破損journalを検証する。捕捉はfake GitHub・使い捨てgit/state・offline runnerを使い、
journal障害と捕捉だけの復旧も検証する。GPU・実agent・networkは不要。
公開は実gitとfake GitHubでbatch、並行main更新、完全一致、拒否、応答喪失を検証する。
merge後の実運用受入は #1817 に残る。

閲覧の受入テストは [test_decision_acceptance.py](../../../tests/unit/agents_ops/test_decision_acceptance.py) と
[test_decision_log_acceptance.py](../../../tests/integration/agents_ops/test_decision_log_acceptance.py) にある。
#1790 を参考にした3異常系、欠測状態、通常委任・親の直接確認・並行分岐・引継ぎ、
[初回実公開PR #1887](https://github.com/Motoki0705/tennis-lab/pull/1887)のコピーと大量記録を検証する。
取得だけの再試行は仮想事例と明示する。当時のCI事例は実際には新規レビューへ進んでおり、
記録されなかった過去の中間判断は欠測のまま残す。実測サイズと検証は #1817 に記録し、
実装merge後のmain公開とGitHub表示は親が確認する。

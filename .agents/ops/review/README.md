# 親issue単位の明示Codexレビュー

親が確定・指示したPR群を、Codexの実行1回で統合レビューする。
PR open/push/CI完了・時間経過から起動するpoll、hook、timerはない。
入力の正本は [parentのreview-request契約](../parent/README.md#レビュー要求契約schema-version-3version-1と2の読取互換)。
担当親自身が、同じ環境由来セッションID・ghアカウント・ローカルstateで実行する。
review/resolve入口はmerge・ラベル操作・子への修正委任を行わない。

| ファイル | 責務 |
|---|---|
| `parent_review.py` | 明示runの受理、収集、Codex実行、記録公開、親による修正確認 |
| `parent_review_policy.py` | 要求・出力・公開記録の検証とcoverageの純粋関数 |
| `review_recovery.py` | 保存final・runtime・元runの検証と採用来歴の照合（AI起動なし） |
| `review_inspection.py` | 親CLIの読み取り専用照合表示、PR別の証拠履歴・入力差・不明範囲 |
| `review_observations.py` | 新runの本文・コメント観測と、意味を判断しない文字列差分 |
| `review_evidence.py` | gateと照合表示が共用する公開済み解消・証拠完了の検査とPython CI取得 |
| `review_locks.py` | runをまたぐPR単位の排他と、未完了executionの照合 |
| `review_interruption.py` | 中断runのcleanup・workspace照合と、承認に使わない終結記録の検証 |
| `review_workspace.py` | PR別headの読み取りworktree、全文diffとプロンプト、先行mergeによるbase前進の実git検証 |
| `merge_gate_policy.py` | 公開証拠の照合、親の確認状態・PRごとのcoverage・依存順・merge判定の純粋関数 |
| `merge_gate.py` | 親の明示コマンドによる最新情報の取得、dry-run、順次merge |
| `review_policy.py` | blocker/major解析、無害化、CI正規化、競合等のgate判定部品 |
| `prompt_template.md` | 親の受入条件・個別diff・依存/インターフェースを1実行で確認する指示 |
| `result_schema.json` | 共通ランナーへ渡す出力schema |
| `../lib/review_tools/` | 固定commit・GitHub GET取得と隔離CPUテスト |

## 実行

review・取得・公開・復旧・明示確認とmerge gateの観測は
[判断記録](../decision_logs/README.ja.md#捕捉元と未対応入口)へ保存する。
捕捉は、このREADMEの実行・再利用・復旧・merge判定を変更しない。

```bash
# 担当親がparent CLIでprepare / request済みのJSONだけを渡す
.venv/bin/python .agents/ops/review/parent_review.py run --request tmp/review-request.json

# 指摘ごとの修正を、Codexを起動せず親が記録する
.venv/bin/python .agents/ops/review/parent_review.py resolve \
  --run parent-1216-32桁hex --pr 1250 --finding pr-1250-1 \
  --fixed-head 40桁SHA --evidence tmp/fix-evidence.json

# 公開途中の失敗・応答喪失から同じrunの公開だけを再開する
.venv/bin/python .agents/ops/review/parent_review.py resume-publication --run parent-1216-32桁hex

# 復旧しない中断を、runtime cleanupとworkspace撤去の確認後に明示終結する
.venv/bin/python .agents/ops/review/parent_review.py close-interrupted \
  --run parent-1216-32桁hex --reason-file tmp/interruption-reason.txt

# parent review-prepare --recover-source-run 元run → review-requestしたJSONを使う（AIは起動しない）
.venv/bin/python .agents/ops/review/parent_review.py recover \
  --request tmp/recovery-request.json --source-run parent-1216-元runの32桁hex \
  --reason-file tmp/recovery-reason.txt

# PRの証拠待ちを、親が同じ4キーの証拠で完了する
.venv/bin/python .agents/ops/review/parent_review.py complete-evidence \
  --run parent-1216-32桁hex --pr 1250 --wait 1 --evidence tmp/evidence.json
```

`fix-evidence.json` は `ci` / `regression_tests` / `resolution` / `reviewed_paths` の4キーだけを持つJSON object。
前3つは非空の文章とし、CIの対象head・結果へのリンク、回帰テストのコマンドと結果、指摘解消と修正範囲の確認理由を書く。
`reviewed_paths` は、親が差分を確認した全変更パスの重複なし・ソート済み配列。
指摘のpath以外に必要な回帰テスト・関連ファイルも明示し、レビューheadから現在headまでの実変更パス（renameの旧/新パスを含む）と完全一致させる。
未記載パス・理由の欠落・旧3キー形式は公開前に拒否する。指摘のpathがnullでも、親による明示確認があれば同じ契約を使える。
CI・テストの実行と証拠の確認は親の責任。このコマンドはその確認を記録し、修正差分は実gitから取得する。
resolve出力のcoverageは当該PRの修正確認に関する参考判定。他PRの最新状態は取得しないため、
gateは必ず全対象の最新スナップショットを別途取得して判定する。
`--fixed-head` は現在のPR headと一致し、branchは元の対象と一致する必要がある。
rebaseやbaseの取込みは下記のファイル単位検証で継承する。それ以外の追加変更は
`--finding changes` と同じ4キーの証拠で親が確認できる。実指摘の解消は指摘ごとに記録する。
修正確認は親と当該PRに公開し、横断指摘は影響するPRごとに確認する。

共通の `--state-dir`（サブコマンドより前）はparent CLIと同じprivate stateを指定する。
既定値は `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/parent-sessions/`。
runの `--agent-exec` はテスト用wrapper指定、`--agent-timeout` は正の有限秒数（既定1800）。
終了コードは0=完全な結果公開または修正確認記録、1=run失敗/未完了/公開失敗、2=入力・担当・整合エラー。
`request_changes` はレビュー完了なので0だが、指摘解消まではmerge可を意味しない。
証拠待ちを含む結果も公開の成功は0であり、論点の完了まではgateが停止する。

## 受理・実行・再送

### runをまたぐPRの排他

run・recover・resume-publication・close-interruptedは、同じprivate state内のrepoとPR番号をキーにした
`review-targets/` のflockを、PR番号順で非待機取得する。その後にrun lock、registry lockを取得する。
PRのleaseは収集前からruntime終了・workspace cleanup・事後照合・公開の終了まで保持し、
重なるPRを含む別runは起動前に拒否する。別PRは同じ親でも並行実行できる。
lock fileはunlinkしない。例外・プロセス終了でflockが解放されても、保存executionが`running`なら新規起動を拒否する。
`published` / `publication_failed` も終了の証明ではない。agent工程の`finished`・`cleanup_complete=true`・
`workspace_cleanup_complete=true` が揃わなければ、下記の明示確認まで同PRの別runを拒否する。
起動0回で`not_started`、またはadoptionの`not_invoked`の場合もcleanupとworkspace撤去の記録を照合する。
新しい準備失敗・adoptionはこの確認を保存する。工程がない旧形式は終了を推定せず、明示確認を要求する。
経過時間・PID・公開成功だけで解放しない。
元runの終了とcleanupを確認し、既存の公開再開条件を満たす場合は同じrunを再開する。
条件を満たせない中断は、下記の明示終結で終了を確認するまで保留する。別runへの自動切替は行わない。

### 明示要求の検査

1. version=3（旧version=1/2も読取可）の要求を厳密に検査する。prepareだけのJSONは拒否する。
   ローカルregistryのactive担当、環境由来ID、openな親、有効な実行根拠、保存済みrun、
   同じrun ID/request digestの公開指示を照合する。
   現在の担当権限と有効な実行根拠を確認する。過去のrequest・実行の担当は来歴として保持し、
   担当交代や本文の文字列変化だけで証拠を失効させない。現在の条件への適用可能性は親が判断する。
2. 実行前に `execution.json` をflock下で保存する。起動回数0の準備失敗は履歴を残して同run IDで明示再試行できる。
   起動後の同run ID再送は既存結果を返し、Codexは再実行しない。異なるスナップショット・古い指示・変更後の対象に再送した場合はエラー。
   他のセッションへの自動振替も行わない。
3. 正式な親子関係とclosing指示で、選んだopen子PRの部分集合を照合する。
   重複・無関係・closed/draft/forkを拒否する。他のPRの追加・更新・closeは対象PRを失効させない。
   head/base先端/base ref/branch/merge-baseとdiffの生バイトSHA-256が固定値と一致すること、
   全PRの情報・CIが取得できたことを確認し、収集中とレビュー後にも取り直す。
4. 呼出元checkoutの `.claude/worktrees/review-<run ID>/` に、各 `pr-<N>/` のdetached head、
   `pr-<N>.diff`、`base-<N>-AGENTS.md`、`context.json`、`request.json` を置く。
   contextは親本文・全コメント・子の要件/依存・PR本文/CI/変更ファイルを含む。
   metadataはPRのcheckout外へ排他的に作成し、PR側のファイル・symlinkを辿って書かない。
   既存workspaceは削除せずエラーにし、今回作ったworktreeは終了時に撤去する。

新runのcontextには `review_observations`（schema_version=1）も保存する。
親と選んだPRのclosing対象子の本文・discussion、PR本文・discussion・inline・review投稿を含む。
既存の指示コメント分類で認識される運用表示は `excluded_records` にIDを分ける。
過去runの観測を後から補完・書換えしない。
5. `agent_exec.sh --agent codex --profile review --mode read-only` を**1回だけ**呼ぶ。
   version 3のmodel/effort/service tier・理由・選択元はrequestのreview_selectionに固定済み。
   起動前に [`../lib/agents.conf`](../lib/agents.conf) のreview方針・許可された組と照合し、runnerへ専用入力で渡す。
   executionとruntimeの選択値、実効model・effortの一致を共通部品で検証する。tier未取得は理由付きで保持する。
   CLIイベント・実効設定・使用量は `agent-runtime/` に保存する（[共通ランナー契約](../lib/README.md)）。
   agentは専用プロセスグループで起動し、timeout/異常終了時は子孫も停止する。
   プロンプトは親の目的・親番号・run ID・固定値・資料とツールの入口だけを含む。
   context・全文diff・schemaは埋め込まない。確認不能ならincompleteとし、追加agentへ分割しない。
6. 出力スキーマは `--json-schema result_schema.json` で固定する。
   [APIの対応範囲](https://developers.openai.com/api/docs/guides/structured-outputs#supported-schemas) に合わせ、
   重複対象の検査は出力後のparserで行う。全PRの `reviewed/not_reviewed` を必須とし、
   severityはblocker/majorのみ。総合判定は全件reviewedかつ指摘なしでapprove、
   指摘ありでrequest_changes、一部未確認でincomplete。不明値・キー過不足・対象違い・
   判定矛盾・パース失敗は失敗。部分収集・途中更新も成功扱いにしない。

run dirはparentの `reviews/<run ID>/` を共有し、固定済みrequest/diffに加えて
`execution.json`、`context.json`、`prompt.md`、`agent_output.md`、`output.json`、`result.json`、コメント原文・agentログを残す。
executionのstatusは `preparation_failed` / `running` / `published` / `publication_failed` / `interrupted_closed`。
新しいexecutionは `stage_version=1` と `stages` を持つ。旧形式は読み取りを維持する。

| 工程 | 記録 |
|---|---|
| `agent` | 起動前/実行中/終了、runtimeのdigest・終了コード、runtimeとworkspaceのcleanup完了 |
| `output` | 保存finalの形式・schema・対象・判定整合の検証結果、finalと解析結果のdigest |
| `postcheck` | 対象と担当権限の再取得・照合結果 |
| `publication` | pending → projections（完了したPR）→ result → published、親コメントID、失敗理由 |

完了イベントと対応しschemaを満たすfinalはruntime非ゼロ終了後も解析し、`output.json` に結果候補を保存する。
通常runの `result.json` はruntime・出力・事後照合を通過した結果、または失敗結果であり、候補とは区別する。
事後照合や公開が失敗してもfinal・解析済み候補を初期化しない。公開失敗でも `result.json` の内容は変えない。
起動回数0の準備失敗は公開せず、明示再試行時に旧資料を `preparation-attempts/` へ保存する。
強制終了後にrunningが残ったら「実行中または中断・未完了」であり、再送で再実行しない。
run lockを取得でき、runtime終了・cleanup・保存結果・事後照合が確認できる公開中断だけは `resume-publication` で再開できる。
起動後の失敗結果も保存・可能なら公開する。GitHub障害なら未公開/部分公開をローカルへ記録して非ゼロ終了する。
復旧は上記の `recover` または `resume-publication` を親が明示実行する。自動再実行は行わない。

### 復旧しない中断runの明示終結

`close-interrupted --run ID --reason-file F` は、担当親の権限・登録済みrequest・非空の理由を検査する。
AIもGitHub書込みも行わず、確認後の記録は元の状態によって分ける。

| 元のexecution | 保存する確認 |
|---|---|
| `running` | ローカルの`interrupted_closed`と`interruption`。未公開の終結で、承認・coverageには使わない |
| `published` / `publication_failed` / `preparation_failed` | `exit_confirmation`だけを追記。status・工程・終了時刻・正式結果・公開状態を維持する |

`exit_confirmation`はleaseの解放確認にだけ使い、gate・coverage・resolve・採用元の検証には使わない。
未確認工程を`finished`へ書き換えず、failed結果や指摘・coverageを昇格させない。
全requestはそのまま残すため、`superseding_reviews`の判定も変えない。
最初に全対象PRのlease、次に対象runの `run.lock` を非待機で取得する。
共通ランナーのdirectoryがあれば`agent-runtime/session.lock`も非待機取得し、確認記録の保存まで保持する。
いずれかを保持中のプロセスがあれば拒否し、経過時間・PID・boot情報から終了を推測しない。

agent起動回数が1、または起動0と`not_started`（adoptionは`not_invoked`）の両方を確認できなければ、
同じrunの `agent-runtime/result.json` が終了状態で `cleanup_complete=true` であることを要求する。
runtimeのrun・出力・finalパスとworkspaceも照合するが、finalファイルの存在・成功・形式検証は要求しない。
runtimeが存在して未完了の場合は、[共通ランナーの停止・cleanup確認](../lib/README.md#状態確認停止中断からの復旧)
が必要な手順としてエラーに出る。runnerが失われた場合はsupervisorの全子孫照合を経て
`agent_exec.sh stop --cleanup-confirmation` で記録し、cleanup確認を手編集で代替しない。
agent未開始を工程と起動回数の両方で確認できる場合はruntimeを要求しない。

runtimeの`result.json`が初回保存前で存在しない場合、共通ランナーのstopもそのファイルを必須読取するため使えない。
親がwrapperと全子孫を含むsupervisorの隔離scopeを照合して回収した後、次の6キーだけのJSONを
`close-interrupted --cleanup-confirmation F`へ渡す。ファイル欠落・既知PIDの消滅だけでは受理しない。

```json
{
  "schema_version": 1,
  "run_id": "parent-1216-対象reviewの32桁hex",
  "runner": null,
  "process": null,
  "cleanup_complete": true,
  "evidence": "親が確認した隔離scope、wrapperと全子孫の回収根拠、対象requestとの対応"
}
```

共通ランナーのcleanup確認validatorを再利用する。native UUIDが未保存なので`run_id`は既知のreview run IDに結び付ける。
`runner`・`process`のnullはidentityが未観測という意味で、プロセス不在の証明ではない。
`evidence`は親による全scope回収の非空の申告であり、CLIがその内容を意味判断することはない。
記録が存在する場合はこの経路で上書き・迂回せず、native IDに一致する共通stopの確認を使う。
確認JSONは終結または終了確認の記録に保存し、runtimeのresultを合成しない。

保存済みrepo設定・現在のrepo・共有git worktree登録で対象workspaceを検査する。
残存ディレクトリ・symlink・worktree登録があれば拒否する。cleanup確認後、対象workspace内の各PR checkoutを
`git worktree remove --force` で撤去し、残ったworkspace metadataを確認して除去する。終結コマンドは削除しない。

終結はexecutionだけを原子的に更新し、`interruption` に理由・実行権限・session・時刻・request digest・
agent開始の可能性・runtime記録のdigestまたは上記の確認JSON・workspace撤去確認を保存する。元の起動回数・工程・request・
final・候補・結果・公開コメントは保持し、正式なfailed結果や採用記録も新設しない。
同じ理由での再送は同じ記録を返し、異なる理由・確認JSONは拒否する。
`interrupted_closed`になった同runの再実行・recover・公開再開は拒否する。
保存finalを採用または公開再開したい場合は、終結する前に既存のrecover・resume-publicationを選ぶ。

終結済みrunは同PRの新runを塞がないが、registryのrequestは残る。gateは
`parent_review_interrupted_closed` で停止し、過去のapproveへ戻らない。
`superseding_reviews` でも通常の後続requestとして残り、より古い証拠の採用を許可しない。
終結runを採用元やcoverageに使わず、他runの未解決指摘・証拠待ち・解消記録も変更しない。

追加AIレビューは `parent review-prepare --reason-file` → `review-request` の新しい明示指示だけで受理する。
後続runは結果の `exception` と人間向けコメントに、通常1回の予算を超える例外として理由・先行runを表示する。

### 保存finalの検証付き採用

新版の要求はprepare時の`--recover-source-run`で元runを固定する。
この参照はrequestのdigestとexecutionへ含め、別の元runへの差替えや新しいagent起動を拒否する。
現在の既定モデルではなく、元runの選択と実効値を検証して採用結果へ表示する。
旧requestは選択理由を補わず従来の証拠として検証する。

`recover` は既存parent CLIで新しくprepare/requestした別runを使用し、元runと同じrepo・親・全PR snapshotを要求する。
`--reason-file` には親の採用理由を書く。元runのrequest・execution・result・final・公開コメントは変更しない。
元runの起動1回、終了したruntimeとcleanup、prompt/schema/runパス・実行先の対応、固定diffのdigest、保存当時の出力schema、
判定/coverage/指摘の整合、現在の対象の再取得を検証する。共通ランナーと同じイベントparserとschema validatorで
`events.jsonl` を再読し、session/turn・実効設定・エラーの記録と、完了イベントの最終メッセージとfinalの内容を照合する。
異なるfinalのコピーや完了後の別メッセージ、形式矛盾、cleanup不明、対象変更は拒否する。
許可するruntime終了理由は `completed`、`runtime_error`、`cli_error` のみ。後二者も完了イベントとfinalが一致し、
終了コードと記録済みエラーが理由に整合するときだけ対象にする。`invalid_output`、`metadata_error`、実効設定・実行先の不一致、
証拠不足、timeout・停止などはschema上有効なfinalがあっても採用しない。
`agent_output.md` が無く `agent-runtime/final.md` だけが残った場合も、上記のイベント照合は必須。
旧runも必要なruntime・schema・final・イベントが揃う場合だけ採用できる。
runningのまま終了確認が不足したrunは採用せず、AIを起動する経路へ自動で切り替えない。

採用元は各対象PRの最新requestである必要がある。registry順で元runより後の全requestを照合し、採用先自身を除く
同じPRのrequestがあればrecoverを拒否する。後続requestの準備中・失敗・incomplete・取消・採用・成功やresolve済みも除外しない。
たとえばAの承認後にBがmajorを指摘した場合、CからAを採用してBの指摘を隠すことはできない。
Bの有効なfinalを採用して指摘を引き継ぎ、採用runでresolveする。Bのresolve後も古いAの採用は再び有効にならない。
公開再開にも同じ制約を適用する。gateは既に公開された採用も再照合し、後続requestがあるPRを
`adoption_source_superseded` で停止する。別PRだけの後続requestはそのPRの採用を妨げない。

採用は `kind=adoption` の新しい正式結果として公開する。指摘は採用runのIDでresolveできる。
gateは元runのregistry/request/実行証拠と採用時のdigestを再照合し、採用runの起動0回と元runの起動1回を区別する。
採用後も親の `review-state --state fixes_verified`、未解決指摘、最新HEADのPython CI、依存関係の検査が必要。
採用元の失敗は来歴として残る。同じ採用runの失敗・未公開・不整合を成功へ読み替えない。

### 証拠待ちの完了

新しいagent出力はPRごとに `evidence_waits: [{question, required_evidence}, ...]` を必須とする。不要なら空配列。
コードの確認は `coverage=reviewed` のまま、判断に必須の証拠が残った論点を記す。未確認コードは従来どおり `not_reviewed`。
総合verdictはコードcoverageと指摘で決まり、証拠待ちは独立したgateの停止条件になる。
CI結論はgateが照合する。CIログ本文の取得失敗や隔離CPUテストの利用不可だけでコード確認全体を未確認にしない。

`complete-evidence --wait <1からの論点番号>` は親が確認した4キー証拠を、現在のPR snapshotと結果digestへ結び付ける。
この場合の `reviewed_paths` は現在のPR全差分（merge-base→head、renameの旧/新パスを含む）と完全一致させる。
指摘のresolveや未確認のコード変更の確認とは独立し、指摘解消やcoverageを代替しない。
公開済みのPR側・親側の記録とローカルのpublishedが揃ってからgateが受理する。同一証拠の再送は重複投稿しない。

## 既存証拠の照合表示

入口は [parent CLIのreview-inspect](../parent/README.md#レビュー証拠の照合表示) 。
prepare前に実行でき、request・registry・公開コメント・担当を変更せず、AIを起動しない。
担当セッションIDは不要。GitHub GETと、必要な固定git objectの取得だけを行う。
照合表示自体は予約でもmerge許可でもなく、実行・復旧・mergeはそれぞれ最新対象を再検査する。

JSONの `prs` は番号順。各要素の `latest_request` はregistry順で最後に対象となったrequestであり、
成功したものだけを選ばない。`history` は古い順に、次の事実を含む。

| 欄 | 内容 |
|---|---|
| `snapshot`, `comparison` | 元snapshotと確認したパス、現在との各フィールドの新旧・一致、head間の変更パスとdiff digest、既存rebase継承検査の結果 |
| `execution`, `runtime`, `result` | 記録された状態・起動回数・工程、runtime終了理由とcleanup、正式結果のcoverage・個別と横断の指摘・証拠待ち |
| `review_selection`, `runtime_settings` | 固定した方針・選択値・理由・指定元と、観測した実効値。旧requestはlegacy_unrecorded、adoptionは元runの検証済み値 |
| `settings_check` | 実効値の確認状態。失敗runの未確認値はunconfirmedとして観測し、成功記録との矛盾は整合エラーにする |
| `publication_errors`, `parent_run_url` | gateと共通の公開照合で得た不一致と親結果への参照。公開検査は元run全体のPR集合に対するもの |
| `followups`, `coverage_check` | 解消・証拠完了の記録、ローカルと両公開コピーの検査、現在snapshotとの一致、共通coverage検査の未確認理由・未解決指摘 |
| `input_changes` | 親・対象子・PRの本文とコメントの差。本文は生文字列と既存requirement_body正規化後の両方を表示 |
| `subsequent_requests`, `superseding_source_requests` | 後続requestと、採用元を置き換える同PRのrequest。失敗や未実行を除外しない |
| `saved_output` | 保存された形式検証済み候補。`candidate_only`と明示し、正式結果やcoverageを置換しない |

PRの `python_ci` は現在HEADのcheck-runと共通CI検査の理由。`coverage_check` はコードと記録の
機械的な照合であり、公開・CI・現在の権限・要件の意味を含むmerge判定ではない。
親が再利用・復旧・証拠完了・resolve・追加レビュー・保留の範囲を判断する。
`recommended_action` や意味上の「再利用可」は返さない。

中断は保存された `execution.status=running` とruntimeの終了・cleanupを別々に表示し、生存は推測しない。
明示終結は `execution.status=interrupted_closed` と `execution.interruption` に表示する。
正式結果を維持した終了確認は`execution.exit_confirmation`に表示し、元の工程・coverage照合も保持する。
終結前の結果が残る場合は `result.retained_only=true` とし、指摘などの履歴を保持する。
終結runの `coverage_check` は `null` とし、保存済みapproveを正式証拠として計算しない。

旧contextに本文だけある場合は本文差だけを比較し、記録されていない子・PRコメントは
`baseline_unavailable`にする。比較元がないことを「変更なし」に置換しない。
取得・形式のエラーは `errors` にscopeとerror_typeを返し、他PRの取得済み事実は保持する。
`publication_unknown_prs` は公開コピーを取得できなかったPR、`projection_errors` は当該PRの
projection照合だけの理由。元run全体の公開エラーと分けて読み、他PRの不明を当該PRの未確認へ置換しない。
収集中のPR・議論・当該親registryの変化も明示する。取得・整合エラー時はJSONを返して終了2、
正常な観測は終了0。指摘・証拠待ち・過去の失敗・比較元欠落は観測対象であり、それだけでは取得エラーにしない。

## 公開記録（schema version 1、#1219の接続契約）

コメント先頭行だけを機械的に読む。

```text
<!-- parent-review:v1 <canonical JSON object> -->
```

JSONはUTF-8/canonical（`sort_keys=True, separators=(",", ":")`、マーカー内は `ensure_ascii=True`）、
`<` / `>` をUnicode escapeする。digestはparentと同じ `ensure_ascii=False` のcanonical JSONのSHA-256。
人間向け本文もHTMLコメントを無害化する。本文中の引用、他login、旧PR単位review markerは証拠にしない。
`records(comments, trusted_login)` は認証中publisherの先頭行だけを読み、不正な記録は例外で拒否する。

`kind=result` の親コメントとローカル `result.json` が同じ内容の正本。キーは以下のみ。

| キー | 内容 |
|---|---|
| `schema_version`, `kind` | `1`, `"result"` |
| `review_run_id`, `request_sha256`, `request` | run ID、prepare manifestのdigest、`review_state`追加前のmanifest全体（親/repo/理由/各PRのhead/base/ref/merge-base/diff/branch） |
| `reviewer`, `profile` | `"codex"`, `"review"` |
| `runtime_settings` | request version 3に必須のキー。成功時はhost検証済みselected/effective、失敗で未確認ならnull。tier未取得は理由付きunavailable、復旧時は元runtimeの値 |
| `started_at`, `finished_at` | timezone付きUTC ISO 8601 |
| `status`, `verdict` | `succeeded/failed`, `approve/request_changes/incomplete` |
| `summary`, `error` | 総合要約、成功ならnull・失敗なら理由 |
| `exception` | 初回null、後続は `{"previous_runs":[run ID...],"reason":"明示理由"}` |
| `cross_findings` | findingに `id=cross-連番`、`prs=[影響するPR番号...]` を追加 |
| `prs` | 全対象の `{number, child_issues, coverage, summary, findings, evidence_waits}`。旧形式のevidence_waits省略も読取可。個別findingのidは `pr-番号-連番` |

findingはidに加えて `{severity,path,line,rule,detail}`。pathはrepo相対POSIXパスまたはnull、lineは正整数またはnull。
PR/子issueとbranchが修正担当の対応先。横断指摘は `prs` の各担当へ振り分ける。
`validate_result` は対象digest・全PR集合・判定/coverage・指摘IDを厳密に検査する。
外側のresult/projection/resolutionと `parent-review:v1` マーカーはversion=1を維持する。
既存の意味・digestを変えず、追加kindと省略可能な証拠待ち欄として拡張するため、外側の版は上げない。
request version 3のruntime_settingsも外側version 1のまま厳密に検証する。
公開・公開再開・照合表示・gateは同じ検証部品でrequest、execution、runtime、公開値を照合する。
利用不能・選択/実効値不一致・継続時の設定不整合は失敗とし、別モデルへ再試行しない。
過去の証拠は保存した方針で検証し、現在のfixed/select設定だけでは失効させない。
version=2/3のauthorizationとrequirementsは、作成時の担当・根拠・本文の観測記録である。
意味のある要件変更と証拠への影響は親が判断し、必要な再検証や確認を記録する。
PRとclosing対象・対象コードの同一性は機械的に照合し、過去のrequest/result/executionを書き換えない。
未知のrequest/basis/policy versionや破損は拒否し、未レビューの空記録へ移行しない。

各PRの `kind=pr` は `{schema_version,kind,review_run_id,parent_issue,pr,request_sha256,
result_sha256,snapshot,coverage,findings,parent_run_url}`。
`result_sha256` は親result全体のdigest、snapshotはprepareの当該PR、findingsは個別＋当該PRに関係する横断指摘。
結果に `evidence_waits` があるPRはprojectionにも同じ配列を含む。旧projectionは省略したまま照合する。
親コメントへのpermalinkからrun全体を読める。

公開は「親に `kind=pending`（schema_version/run ID/request digestだけ）を作る →
全PRへprojectionを投稿 → 親コメントをresultへ更新」の順。pendingやPRコメント単独は成功の証拠ではない。
途中失敗では工程の失敗をローカルへ保存する。`resume-publication` は認証loginの先頭行recordを再取得し、
一致するpending/projection/resultを再利用して欠けた書込みだけを行う。投稿の応答喪失も同じ照合で復旧し、
同runの別内容・重複・先行failed記録があれば停止する。公開済みコメントは上書きせず、pendingだけを正式結果へ更新する。
gateは成功resultと全PRのprojection/digest、
ローカルexecutionのpublishedを照合し、同runの失敗記録・公開不整合があれば停止する。
投稿応答喪失も自動再レビューの理由にしない。

`kind=adoption` はresultと同じ結果キーに `mode,source_run,reason,agent_invocations,validation` を加える。
現在のmodeは `recover` のみで、`agent_invocations=0`。
`validation` は `{source_request_sha256,source_execution_sha256,source_result_sha256,runtime_sha256,events_sha256,final_sha256,output_sha256,current_prs}`。
各digestは元runの実ファイルと再解析した候補を拘束し、`current_prs` は再取得して一致した採用対象。
`events_sha256` の無い旧採用記録の読取は維持するが、gateは現行の検証証拠との完全一致を要求するため受理しない。
証拠を満たさない旧採用は正式結果へ読み替えない。採用元が後続requestで置き換わっている場合は、
新しい明示requestによるレビューで最新の実行証拠を作る必要がある。
同じ共通の結果・projection・resolve・gate経路に接続する。将来のreuseはmode固有の検証を追加する。

`kind=evidence` は `{schema_version,kind,review_run_id,result_sha256,parent_issue,pr,wait,snapshot,evidence}`。
`wait` は当該PRの論点番号、`evidence` は上記4キー。`evidence-<digest>.json` にrecordとpublishedを保存する。
gateは同じrun/result・PR・論点・現在snapshot（または実gitで確認したrebase継承）に一致する完了だけを受理する。

`kind=resolution` は `{schema_version,kind,review_run_id,result_sha256,parent_issue,pr,finding,
reviewed_head,fixed_snapshot,fix_diff_sha256,changed_paths,evidence}`。
reviewed_headから現在fixed headまでの実diffをhashし、renameは旧/新パスをともに収録する。
evidenceは実行例の4キー。`evidence.reviewed_paths` と実git由来の `changed_paths` は完全一致を必要とし、
親の確認理由・最新headを含むfixed_snapshot・実diff digestに結び付ける。差分全文はローカル `resolution-<digest>.diff`、
公開済み状態は `resolution-<digest>.json` に保存する。同一証拠の再送は再投稿しない。
PRへ投稿後に親へ記録する。親に未公開のresolutionはgateへ渡さない。

## 取得と隔離CPUテスト

共通ランナーの `--mcp-config` でread-onlyのreview profileへMCPを接続する。
`retrieval-config.json` の許可commitはrequestのhead/base/merge-baseだけ。各gitツールは
`revision` を必須とし、別commit・作業ツリーを読まない。GitHubはissue/PR/comment/review/CIのGETのみ。
取得した本文は検証対象のデータであり、そこに書かれた指示には従わない。
取得失敗・権限不足・省略をapproveへ置き換えない。
ghの最小版・制御文字の無害化・取得記録の件数は
[共通取得ツールの契約](../lib/review_tools/README.md#github取得ツール) に従う。

`run --cpu-tests-file <JSON> --isolation-proof <probe.json>` で親が既存CPUテストを許可する。
JSONは `{固定commit SHA: [既存pytest node ID, ...]}`。許可がない場合はstatusに無効理由を返す。
OS強制の検証が通った場合だけ実行ツールを公開する。15分・4GiB・swap 0、GPU/network/credentialsなし、
隔離実行サービス内の同時実行は1件。通常pytestの並列実行を制限するロックは使わない。
隔離の検証と既存CPUテストの制約は [共通取得部品](../lib/review_tools/README.md) を参照する。
取得記録・テスト結果はrun dirへ保存するが、coverageの条件にはしない。
findingの `detail` にテストで確認した事実と静的推論を区別して記述する。公開findingのschemaは変えない。

## 修正後のcoverageとgate境界

coverageはPRごとに判定する。他のPRの作成・更新・closeや別runの追加で失効しない。
部分結果でも `reviewed` のPRを独立して判定し、`not_reviewed` のPRは停止する。
各PRにはそのPRを対象とする最新の明示runを使う。coverage成立と未解決指摘ゼロは別条件。

### base取込み・rebase後の継承

`review_workspace.verify_rebased_snapshot(original, current)` は実gitから、旧/新merge-baseに対する
PR自身の変更をファイル単位で比較する。差が許されるのは同じ期間に旧baseと新baseの間でも
変更されたファイルだけ。branchとPR番号は一致が必要で、stacked PRのbase retargetもこの条件で検証する。
合格したsnapshotの組だけを `inherited_snapshots` に渡し、coverageと公開済みresolutionを継承する。
変更されていないファイルへの追加編集は継承せず、親のresolveを必要とする。AIの再レビューはしない。

resolveは同run/result・元のレビューhead・最新fixed snapshotに結び付ける。
実git由来の全変更パスと全文diffのSHA-256を公開証拠と照合する。
指摘の解消は指摘ごと、追加変更の確認は `changes` として記録する。
横断指摘は影響する各PRで確認する。未知のschema・不完全な証拠は停止する。

## merge gate

担当親自身が、parent CLI・reviewと同じセッション環境、ghアカウント、private stateで明示実行する。
GitHub API・CI取得・merge通信中はregistry lockを解放し、戻ると担当親の状態変更を照合する。
gateのGETは鮮度を優先して再取得し、通信待ちで他の親を止めない。
timer・poll・hook・agent呼出はない。判定は `merge_gate_policy.evaluate_parent_gate(MergeInput)`、
I/Oは `merge_gate.py` に分離する。共通引数はサブコマンドより前に置く。

公開ログの専用入口はこの通常gateを緩めず、形式ごとの全差分検査で限定例外を扱う。
[workerの3ファイル](../worker_logs/README.md#専用prと限定merge例外)と
[判断記録・再生成README](../decision_logs/README.ja.md#限定公開例外と復旧)は別の検査であり、
コードや規則が混在するPRには適用しない。

```bash
# 未設定または0: 判定とJSON表示だけ。GitHubの更新・mergeは行わない
AGENT_AUTO_MERGE=0 .venv/bin/python .agents/ops/review/merge_gate.py --parent 1216 merge

# 人間の有効化判断後、呼出元の親セッションを起動したローカルshellの環境へ設定
export AGENT_AUTO_MERGE=1
.venv/bin/python .agents/ops/review/merge_gate.py --parent 1216 merge

# 停止: 次の明示実行からdry-runに戻す（unsetでも同じ）
export AGENT_AUTO_MERGE=0
```

`AGENT_AUTO_MERGE` は未設定/`0`/`1`だけを受け付け、空文字を含む他の値はI/O前にエラーにする。
設定は呼出元環境だけで行い、repo内へ有効値を書き込まない。既に動いているコマンドの停止には、
そのプロセスを中断する。試行前に保存した記録は残るため、再送でmergeを繰り返さない。
#1216の人間との追加合意に基づき、#1219のmerge後に親がこの環境変数を`1`へ変更する。
既存の全体レポートtimerやGitHub標準のAllow auto-mergeとは別の仕組みである。

gateは以下をすべて検査し、各PRを理由コード付きの `merge` / `blocked` / `needs_human` とする。

- 最新の正式な直属子とPRのclosing指示、review対象集合・子の割当、既知の作成agent、
  open・draft/fork・担当branchの整合。同じ子のopen PRが複数あれば停止する。
- repository情報から取得した既定branchとREST PRの `base.ref` が一致すること。
  不一致は `base_not_default_branch` で停止し、baseは自動で付け替えない。既存のmerge済み対象も確認する。
- 各PRの最新の明示runのローカルresult/request/executionと、同じgh loginによる親result・全対象PRの
  projectionの内容/digest/permalink。executionはpublished・agent実行1回・保存済みrequestに対する来歴の整合が必須。
  pending、同runの失敗結果、他人のコメント、引用、未公開・部分公開は通さない。
- 親の公開review状態が `fixes_verified` または `complete`。結果受領だけ・修正中・中断・
  `needs-human` label（親/子/PR）・子の停止/依存保留ではmergeしない。
  確認は対象runへ結び付け、担当交代や記録revisionだけで失効させない。
  現在の目的・受入条件への適用可能性は親が判断する。親issue全体の完了は要求しない。
- #1218の `evaluate_coverage` と未解決指摘。対象集合等の全体不整合は全件停止するが、
  他PRだけのcoverage不成立や未解決指摘は独立した当該PRの判定に持ち込まない。
  merge済み対象も集合に残し、GitHubのmerge済みhead・branch・base ref/repoを確認済みsnapshotと照合する。
  snapshotはレビュー時、gateによるmerge試行済みならその直前の確認時の値を据え置く。
  resolveは親/PR両方への公開・ローカルpublished・保存差分digestを確認する。
  現headへのresolveは、実gitのレビューhead→現head差分のdigest/パスにも一致させる。
- 競合・mergeable不明は停止する。CI取得時のheadも最新snapshotと一致させる。
  対象HEADの `Python` check runが完了し、結論が `success` であることを必須とする。
  commitを指定したREST `commits/<head>/check-runs?check_name=Python&filter=latest` を全ページ取得し、
  各runの `head_sha` も照合する。既存のCI正規化と共通の時刻検証を使い、開始時刻が最新のrunで判定する。
  時刻なし・同時刻の候補はすべて成功が必要で、古い成功では代用しない。再実行の成功は古い失敗を置き換える。
  欠落は `ci_python_missing`、別HEADは `ci_python_head_mismatch`、未完了は `ci_python_pending`、
  失敗・取消等は `ci_python_<conclusion>` で停止する。`neutral` / `skipped` も成功として通さない。
  同名のStatusContextやlabelerの成功では代用しない。
  CI取得失敗・不正な応答・不明なCI結論は `input_or_collection_error` とし、未実行に読み替えない。
- 子本文の最新依存欄を使い、依存先子がclosed・`state_reason=completed`（`task_status=completed`）であり、
  その子を閉じるPRが既定branchへmerge済みであることを求める。PR一覧は親子issueに関連する全状態・全ページを取得する。
  closing PRがopen・merged・closed未mergeのいずれにも存在しない完了子は、調査だけの成果として依存を満たす。
  判定根拠はJSONの `dependency_statuses` に子番号ごとに記録し、充足は `completed_without_pr` または
  `merged_into_default_branch`。停止理由は `dependency_<N>_not_completed`、`dependency_<N>_open_pr`、
  `dependency_<N>_unmerged`、`dependency_<N>_base_not_default_branch` で区別する。既定branchへのmerge済みPRが
  あってもopen PRが残れば停止する。PRの状態・merge先が不明なら `dependency_<N>_pr_state_unknown` /
  `dependency_<N>_pr_base_unknown`、issueの完了理由が不明・取得失敗・既定branchが不明なら
  `input_or_collection_error`（終了コード2）で停止する。
  循環・親の外への依存も拒否する。状態の正本と停止/再開は [parent](../parent/README.md)。

依存順で判定し、CI待ちのPRは停止する。CIを含む全条件を満たす独立PRは順次進める。
実merge直前に全対象のhead/base/merge-base/diff・CI・競合・coverage・親状態を再取得し、
`gh pr merge --merge --match-head-commit <確認済みhead>` を一度だけ実行する。
各merge後も残りを再取得し、[base取込み・rebase後の継承](#base取込みrebase後の継承)を検証する。
確認済みのmerge済みheadが一致しなければ `merged_head_mismatch` を表示する。

merge直前にflock下で `merges/<repo digest>/pr-<N>-<head>.json` をprivate stateへ保存する。
この試行記録の `snapshot` に確認済みhead/base/merge-base/diff/ref/branchを保存し、次回の明示実行でも保持する。
同repo/PR/headへの試行済み記録はrunをまたいで保持する。異常終了・応答喪失後も同headを再試行しない。
remoteがmerge済みなら二重mergeせず、対象issueのclosing完了を確認する。
収集・割当照合・完了確認の対象はreview-prepare/collectと同じで、直属子があれば子、
子のない単独issueではそのissue自身を閉じるPRを扱い、不要な親子関係を作らない。
merge直後もremoteのmergeと対象issueのclosed/completedを確認する。子のある親だけ既存の
`reconcile` / `sync_progress` へ接続し、単独issueの本文を親用進捗へ書き換えない。
未close・確認不能は `merge_unconfirmed` として残す。親がGitHubとjournalを照合して復旧する。
旧試行記録を削除しての自動再試行はしない。

JSONの `results` が各PRの最終状態、`evaluations` が順次判定記録、`snapshot` が確認したhead/base/diff。
各判定の `ci_checks` はCI観測値（check名と `success` / `pending` / `failure`）、`ci_head_sha` は取得応答のhead。
`snapshot.head_sha` との不一致は `head_changed_during_ci` で停止する。
CIが無い場合は空配列、closed/merge済みPRの再確認でCIを取得しない場合は `null`。
合否には上記のHEAD付きcheck runを使う。rollupの観測値だけでは成功と判定しない。
merge直前や先行merge後も再取得し、未完了なら理由付きで返す。コマンド内にCI待ちループや自動再試行は持たない。
`auto_merge` は実効フラグ、`agent_invocations` は常に0。
終了コードは0=全件merge可/merge確認（dry-runを含む）、1=停止/未完了あり、2=入力・取得・整合エラー。
2でも成功や再レビューへ置き換えず、stderrのblocked記録を確認する。

## CI部品と人間の制御

変更パスに基づく人間承認は不要。人間の制御は呼出元の `AGENT_AUTO_MERGE=1`、
親/子/PRの `needs-human` ラベルによる停止、親の要件合意として維持する。

`normalize_rollup` / `CheckState` は全体調査も再利用する。
workflow名＋check名ごとの最新runへ畳み、時刻なし/同時刻は最も悲観的な状態を採用する。
この観測値の正規化ではSUCCESS/NEUTRAL/SKIPPEDを `success` とする。不明値はエラー。
`evaluate_gate(GateInput)` はdraft/fork・競合とレビューを判定する部品である。
通常PRの必須CIは `merge_gate_policy.python_ci_errors` が別途厳密に判定する。
固定名の集約job `Python` だけを必須とするため `REQUIRED_CHECK` 定数で明示し、通常PR用の一覧ファイルは持たない。
`review` 引数には呼出側が親run/coverage/解消証拠から確認した `ReviewOutcome` を渡す。
専用memory PRの必須CI一覧は [ops-report/required_checks.txt](../ops-report/required_checks.txt) で管理し、
その限定経路のCI要件は維持する。通常PRのmerge gateはこの一覧を読まない。

作成agentのbranch prefixは `claude/` / `codex/`、補助labelは `agent:claude` / `agent:codex`。
`detect_author_agent` は矛盾をエラーにするが、親の明示対象を作成agentで自動選別しない。

## 旧systemd unitの撤去（親が実施）

旧 `cross_review.py` と `tennis-lab-cross-review.{service,timer}` は削除済み。
installerは導入先に残る旧unit（dangling symlink・wants内も含む）を検出すると、
リンク・daemon-reload・enableより前に撤去手順を示して失敗する。旧source unitも再導入しない。
旧週次レポートunitも残っている場合は [ops-reportの導入手順](../ops-report/README.md) で撤去する。
ホストで次を実施し、結果を親issueへ記録する。

```bash
systemctl --user disable --now tennis-lab-cross-review.timer
systemctl --user stop tennis-lab-cross-review.service
systemctl --user disable tennis-lab-cross-review.service
rm -f ~/.config/systemd/user/tennis-lab-cross-review.{service,timer}
rm -f ~/.config/systemd/user/timers.target.wants/tennis-lab-cross-review.timer
systemctl --user daemon-reload
systemctl --user is-active tennis-lab-cross-review.timer tennis-lab-cross-review.service
systemctl --user is-enabled tennis-lab-cross-review.timer tennis-lab-cross-review.service
systemctl --user list-timers --all
# main checkout更新後、旧unitが復活せず全体レポートtimerが残ることを確認
.agents/ops/install_units.sh --dry-run
.agents/ops/install_units.sh
```

他のwants/aliasに旧unit名があれば、installerが示すパスも確認して撤去する。
is-active/is-enabledはinactive/not-found/disabled等の非ゼロを期待する。OS再起動後も同じ状態を確認する。
3日周期の全体レポートtimerは撤去対象ではない。旧changes-requested等のlabelは新入口では使用しない。

テスト: `.venv/bin/python -m pytest tests/e2e/agents_ops -n 4 -q`。
fake gh・fake agent_exec・一時state・実gitを使い、実agent/GitHub書込み/ホストsystemd操作は行わない。

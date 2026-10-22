# 親issue単位の明示Codexレビュー

親が確定・指示したPR群を、Codexの実行1回で統合レビューする。
PR open/push/CI完了・時間経過から起動するpoll、hook、timerはない。
入力の正本は [parentのreview-request契約](../parent/README.md#レビュー要求契約schema-version-2version-1読取互換)。
担当親自身が、同じ環境由来セッションID・ghアカウント・ローカルstateで実行する。
review/resolve入口はmerge・ラベル操作・子への修正委任を行わない。

| ファイル | 責務 |
|---|---|
| `parent_review.py` | 明示runの受理、収集、Codex実行、記録公開、親による修正確認 |
| `parent_review_policy.py` | 要求・出力・公開記録の検証とcoverageの純粋関数 |
| `review_workspace.py` | PR別headの読み取りworktree、全文diffとプロンプト、先行mergeによるbase前進の実git検証 |
| `merge_gate_policy.py` | 公開証拠の照合、親の確認状態・PRごとのcoverage・依存順・merge判定の純粋関数 |
| `merge_gate.py` | 親の明示コマンドによる最新情報の取得、dry-run、順次merge、共有基盤免除の記録 |
| `review_policy.py` | blocker/major解析、無害化、CI正規化、required checks・共有パス・競合等のgate判定部品 |
| `prompt_template.md` | 親の受入条件・個別diff・依存/インターフェースを1実行で確認する指示 |
| `shared_paths.txt` / `required_checks.txt` | 共有基盤パス / 必須CI check名 |

## 実行

```bash
# 担当親がparent CLIでprepare / request済みのJSONだけを渡す
.venv/bin/python .agents/ops/review/parent_review.py run --request tmp/review-request.json

# 指摘ごとの修正を、Codexを起動せず親が記録する
.venv/bin/python .agents/ops/review/parent_review.py resolve \
  --run parent-1216-32桁hex --pr 1250 --finding pr-1250-1 \
  --fixed-head 40桁SHA --evidence tmp/fix-evidence.json
```

`fix-evidence.json` は `ci` / `regression_tests` / `resolution` / `reviewed_paths` の4キーだけを持つJSON object。
前3つは非空の文章とし、CIの対象head・結果へのリンク、回帰テストのコマンドと結果、指摘解消と修正範囲の確認理由を書く。
`reviewed_paths` は、親が差分を確認した全変更パスの重複なし・ソート済み配列。
指摘のpath以外に必要な回帰テスト・関連ファイルも明示し、レビューheadから現在headまでの実変更パス（renameの旧/新パスを含む）と完全一致させる。
未記載パス・理由の欠落・旧3キー形式は公開前に拒否する。指摘のpathがnullでも、親による明示確認があれば同じ契約を使える。
CI・テストの実行と証拠の確認は親の責任。このコマンドはその確認を記録し、修正差分は実gitから取得する。
resolve出力のcoverageは当該PRの修正確認に関する参考判定。他PRの最新状態は取得しないため、
gateは必ず全対象の最新スナップショットを別途取得して判定する。
`--fixed-head` は現在のPR headと一致し、レビューheadの子孫でなければならない。
base ref・merge-base・branchの変更はresolveでも受理しない。base先端だけの前進は後述の先行merge証拠で検証する。
修正確認は親と当該PRに公開し、横断指摘は影響するPRごとに確認する。

共通の `--state-dir`（サブコマンドより前）はparent CLIと同じprivate stateを指定する。
既定値は `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/parent-sessions/`。
runの `--agent-exec` はテスト用wrapper指定、`--agent-timeout` は正の有限秒数（既定1800）。
終了コードは0=完全な結果公開または修正確認記録、1=run失敗/未完了/公開失敗、2=入力・担当・整合エラー。
`request_changes` はレビュー完了なので0だが、指摘解消まではmerge可を意味しない。

## 受理・実行・再送

1. version=2（旧version=1も読取可）の要求を厳密に検査する。prepareだけのJSONは拒否する。
   ローカルregistryのactive担当、環境由来ID、openな親、有効な実行根拠、最新run、
   同じrun ID/request digestの公開指示を照合する。
   公開指示の担当generation・根拠の種別/digestも一致させ、引継ぎ前の指示を流用しない。
   自律根拠の入力が変われば起動・結果の受理・修正確認・mergeを停止する。
   旧version=1はgeneration/agreement digestで人間合意と照合し、新たな自律根拠へ読み替えない。
2. 実行前に `execution.json` をflock下で保存する。同run IDの再送は既存結果を返し、
   Codexは再実行しない。異なるスナップショット・古い指示・変更後の対象に再送した場合はエラー。
   他のセッションへの自動振替も行わない。
3. 正式な親子関係とclosing指示で、その時点の全open子PR集合を照合する。
   重複・欠落・対象追加・無関係・closed/draft/forkを拒否する。
   head/base先端/base ref/branch/merge-baseとdiffの生バイトSHA-256が固定値と一致すること、
   全PRの情報・CIが取得できたことを確認し、収集中とレビュー後にも取り直す。
4. 呼出元checkoutの `.claude/worktrees/review-<run ID>/` に、各 `pr-<N>/` のdetached head、
   `pr-<N>.diff`、`base-<N>-AGENTS.md`、`context.json`、`request.json` を置く。
   contextは親本文・全コメント・子の要件/依存・PR本文/CI/変更ファイルを含む。
   metadataはPRのcheckout外へ排他的に作成し、PR側のファイル・symlinkを辿って書かない。
   既存workspaceは削除せずエラーにし、今回作ったworktreeは終了時に撤去する。
5. `agent_exec.sh --agent codex --profile review --mode read-only` を**1回だけ**呼ぶ。
   model/effort/service tierは [`../lib/agents.conf`](../lib/agents.conf) の `REVIEW_CODEX_*`。
   agentは専用プロセスグループで起動し、timeout/異常終了時は子孫も停止する。
   合計20万文字を超えるdiffはプロンプトを一部に限定し、全文ファイルの読取を明示する。
   大きな本文もcontext全文を参照させる。確認不能ならincompleteで失敗し、追加agentへ分割しない。
6. 出力スキーマは `prompt_template.md`。全PRの `reviewed/not_reviewed` を必須とし、
   severityはblocker/majorのみ。総合判定は全件reviewedかつ指摘なしでapprove、
   指摘ありでrequest_changes、一部未確認でincomplete。不明値・キー過不足・対象違い・
   判定矛盾・パース失敗は失敗。部分収集・途中更新も成功扱いにしない。

run dirはparentの `reviews/<run ID>/` を共有し、固定済みrequest/diffに加えて
`execution.json`、`context.json`、`prompt.md`、`agent_output.md`、`result.json`、コメント原文・agentログを残す。
executionのstatusは `running` / `published` / `publication_failed`。
強制終了後にrunningが残ったら「実行中または中断・未完了」であり、再送で再実行しない。
失敗結果も保存・可能なら公開する。GitHub障害なら未公開/部分公開をローカルへ記録して非ゼロ終了する。
復旧時は親が旧プロセス・workspace・ローカル/公開記録を確認し、必要なら新しい理由とrun IDで指示する。

追加AIレビューは `parent review-prepare --reason-file` → `review-request` の新しい明示指示だけで受理する。
後続runは結果の `exception` と人間向けコメントに、通常1回の予算を超える例外として理由・先行runを表示する。

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
| `started_at`, `finished_at` | timezone付きUTC ISO 8601 |
| `status`, `verdict` | `succeeded/failed`, `approve/request_changes/incomplete` |
| `summary`, `error` | 総合要約、成功ならnull・失敗なら理由 |
| `exception` | 初回null、後続は `{"previous_runs":[run ID...],"reason":"明示理由"}` |
| `cross_findings` | findingに `id=cross-連番`、`prs=[影響するPR番号...]` を追加 |
| `prs` | 全対象の `{number, child_issues, coverage, summary, findings}`。個別findingのidは `pr-番号-連番` |

findingはidに加えて `{severity,path,line,rule,detail}`。pathはrepo相対POSIXパスまたはnull、lineは正整数またはnull。
PR/子issueとbranchが修正担当の対応先。横断指摘は `prs` の各担当へ振り分ける。
`validate_result` は対象digest・全PR集合・判定/coverage・指摘IDを厳密に検査する。
外側のresult/projection/resolutionと `parent-review:v1` マーカーはversion=1を維持する。
新requestのversion=2に含むauthorizationもrequest/result digestへ入り、coverage・修正確認を同じ根拠と担当世代へ結び付ける。
requirementsに固定した親・対象子の要件本文digestも起動・状態遷移・修正確認・mergeで再照合する。
進捗更新は除外し、要件変更は親による結果の採用だけでは旧reviewへ取り込めない。
親の要件変更は全件、子の要件変更はその子を含むPRのcoverageを止める。runの進捗記録は子のcoverageを承認しない。
修正確認は当該PRの子を照合し、別の子の停止/変更を独立PRの確認へ持ち込まない。
旧公開記録は書き換えず、保存済みrequest/result/executionと元のdigestで照合する。
未知のrequest/basis/policy versionや破損は拒否し、未レビューの空記録へ移行しない。

各PRの `kind=pr` は `{schema_version,kind,review_run_id,parent_issue,pr,request_sha256,
result_sha256,snapshot,coverage,findings,parent_run_url}`。
`result_sha256` は親result全体のdigest、snapshotはprepareの当該PR、findingsは個別＋当該PRに関係する横断指摘。
親コメントへのpermalinkからrun全体を読める。

公開は「親に `kind=pending`（schema_version/run ID/request digestだけ）を作る →
全PRへprojectionを投稿 → 親コメントをresultへ更新」の順。pendingやPRコメント単独は成功の証拠ではない。
途中失敗では親へfailed resultを追記する。gateは成功resultと全PRのprojection/digest、
ローカルexecutionのpublishedを照合し、同runの失敗記録・公開不整合があれば停止する。
投稿応答喪失も自動再レビューの理由にしない。

`kind=resolution` は `{schema_version,kind,review_run_id,result_sha256,parent_issue,pr,finding,
reviewed_head,fixed_snapshot,fix_diff_sha256,changed_paths,evidence}`。
reviewed_headから現在fixed headまでの実diffをhashし、renameは旧/新パスをともに収録する。
evidenceは実行例の4キー。`evidence.reviewed_paths` と実git由来の `changed_paths` は完全一致を必要とし、
親の確認理由・最新headを含むfixed_snapshot・実diff digestに結び付ける。差分全文はローカル `resolution-<digest>.diff`、
公開済み状態は `resolution-<digest>.json` に保存する。同一証拠の再送は再投稿しない。
PRへ投稿後に親へ記録する。親に未公開のresolutionはgateへ渡さない。

## 修正後のcoverageとgate境界

`evaluate_coverage(result, current_snapshots, resolutions, changed_paths, *, fix_diff_digests, base_advances=())` はI/Oのない純粋関数。
current_snapshotsは全run対象のスナップショット、changed_pathsは**レビューheadから最新headまでの実git差分のパス**を渡す。
必須keyword引数 `fix_diff_digests` は同じ差分の生バイトSHA-256を `{PR番号: digest}` で渡す。
`review_workspace.full_diff(reviewed_head, latest_head)` をhashし、公開resolutionのdigestをそのまま転記しない。
修正のないPRは両mapから省略できるが、修正のあるPRでどちらか欠ければ確認は成立しない。
返り値は `Coverage(covered, reasons, unresolved)`。unresolvedは `(PR番号, finding ID)` の組。

- failed/incomplete・not_reviewed・PRの追加削除・base ref/merge-base/branch変更はcoverage不成立。
- head/diffが変わったPRは、同じrun/result・レビューheadに対する親の修正確認を必要とする。
  fixed_snapshotは現在のhead/merge-base/diff/ref/branchと一致し、実fix diff digestと全変更パスも記録と一致しなければならない。
  親が `reviewed_paths` に明示した回帰テスト等は受理し、記録外のパス・digest不一致・古いheadの証拠は拒否する。
- 修正が追加pushされたら、以前のheadへの確認を流用せず、現headで各指摘の証拠を再記録する（AI再実行なし）。
- coverage成立と未解決指摘ゼロは別条件。横断指摘は影響する各PRで確認する。

### 同じrunの先行mergeによるbase先端の前進

base先端だけが進み、head・merge-base・diff digest・base ref・branchがレビュー時またはresolve記録時と同一なら、
同じrunの**他の対象PR**のmergeで説明できる場合に限りcoverageと解消確認を継承する。指摘ゼロのPRでもresolveや追加Codex実行は不要。

`review_workspace.verify_base_advances(result, current_snapshots, merged_prs)` が正規化証拠
`BaseAdvance(pr, base_ref, before_sha, after_sha)` のtupleを返す。`pr` はmerge済みの先行対象PR、
`before_sha → after_sha` は実gitのfirst-parent上の1辺、after_shaはGitHubのmerge commit。
current_snapshotsは前進を検査する未merge PRの最新snapshot（対象の部分集合でよい）、merged_prsは
呼出側が同じrepoのGitHub `GET /repos/{repo}/pulls/{n}` から直接得たrun対象PRのraw objectを渡す。
コメント本文やPRが提供したJSONをGitHub応答として渡さない。`merged is True`・対象番号・head branch・base ref/repo・
`merge_commit_sha` を照合し、元のbaseから最新baseまでのfirst-parent上の**全commit**をそのmerge commit群で説明する。
未説明commit、巻き戻し、別base・対象外PR、未merge時のテスト用merge commit、履歴の不足は例外で停止する。
複数commitのrebase merge等も、全commitをこの証拠で説明できなければ停止する。

gateはこのtupleを `base_advances` に渡す。純粋関数は別対象PRによる連続鎖だけを受理し、
originalまたはfixed_snapshotから最新baseまで欠けた辺がないことを確かめる。証拠省略時は従来どおりbase先端変更で停止する。
gateには全run対象を残し、先行merge済みPRはGitHub上のmerge済みheadと確認済みheadの一致を検証してから確認済みsnapshotを据え置く。
残るPRの最新snapshot・必須CI・競合・依存関係は各merge後に再取得する。merge済みPRを集合から削除したり、
未merge PRに過去のsnapshotを渡したりしてcoverage判定を迂回しない。

**PR自身へmainをmerge/rebaseしてmerge-baseやdiffを変えた場合は継承しない。** この版ではresolveによる追従承認経路を設けず、
親が理由を付けた新しい明示review要求を作る。retargetや未レビュー差分も同様に停止し、自動再レビューは起動しない。

未知の入力スキーマは例外で拒否する。gateは例外も停止として扱う。
最新head/CI・必須checks・競合・共有パス・人間判断・依存順の照合とmerge実行は下記のmerge gateが担う。
旧PR単位のCodex承認マーカーは使わない。

## merge gate

担当親自身が、parent CLI・reviewと同じセッション環境、ghアカウント、private stateで明示実行する。
timer・poll・hook・agent呼出はない。判定は `merge_gate_policy.evaluate_parent_gate(MergeInput)`、
I/Oは `merge_gate.py` に分離する。共通引数はサブコマンドより前に置く。

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
このフラグはmergeサブコマンドの制御であり、`shared-permission` は明示的に合意コメントを記録する別操作。
設定は呼出元環境だけで行い、repo内へ有効値を書き込まない。既に動いているコマンドの停止には、
そのプロセスを中断する。試行前に保存した記録は残るため、再送でmergeを繰り返さない。
#1216の人間との追加合意に基づき、#1219のmerge後に親がこの環境変数を`1`へ変更する。
既存の全体レポートtimerやGitHub標準のAllow auto-mergeとは別の仕組みである。

gateは以下をすべて検査し、各PRを理由コード付きの `merge` / `blocked` / `needs_human` とする。

- 最新の正式な直属子とPRのclosing指示、review対象集合・子の割当、既知の作成agent、
  open・draft/fork・担当branchの整合。同じ子のopen PRが複数あれば停止する。
- 最新の親runのローカルresult/request/executionと、同じgh loginによる親result・全対象PRの
  projectionの内容/digest/permalink。executionはpublished・agent実行1回・現在の担当/実行根拠との一致が必須。
  pending、同runの失敗結果、他人のコメント、引用、未公開・部分公開は通さない。
- 親の公開review状態が `fixes_verified` または `complete`。結果受領だけ・修正中・中断・
  `needs-human` label（親/子/PR）・子の停止/依存保留ではmergeしない。
  runの実行根拠digest・担当generationを照合し、無関係な後続の提案合意や同じ根拠の再記録では
  既存の確認と共有基盤の許可を失効させない。親全体の明示的な `checking`、合意内容の変更、
  引継ぎ後は古い確認を流用しない。親issue全体の完了は要求しない。
  新しい委任/結果のauthorizationと子要件digestも照合する。根拠変更後の既存成果は親がrecord-resultで
  現在の要件へ明示的に結び付けるまで使わず、旧レビュー自体は新しい理由/runで取り直す。
- #1218の `evaluate_coverage` と未解決指摘。対象集合等の全体不整合は全件停止するが、
  他PRだけのcoverage不成立や未解決指摘は独立した当該PRの判定に持ち込まない。
  merge済み対象も集合に残し、GitHubのmerge済みhead・branch・base ref/repoを確認済みsnapshotと照合する。
  snapshotはレビュー時、gateによるmerge試行済みならその直前の確認時の値を据え置く。
  resolveは親/PR両方への公開・ローカルpublished・保存差分digestを確認する。
  現headへのresolveは、実gitのレビューhead→現head差分のdigest/パスにも一致させる。
- `required_checks.txt` の全必須checkが存在し、rollup全件に失敗・未完了がないこと。
  CI取得時のheadも一致させる。競合・mergeable不明は停止する。
- `shared_paths.txt` の一致は既定で `needs_human`。下記の明示合意だけで免除できる。
- 子本文の最新依存欄を使い、依存先子のPRが実mergeされ、その子がclosed/completedになるまで保留する。
  循環・親の外への依存も拒否する。状態の正本と停止/再開は [parent](../parent/README.md)。

依存順で判定し、他の独立PRのCI待ちで合格済みPRを止めない。
実merge直前に全対象のhead/base/merge-base/diff・CI・競合・coverage・親状態を再取得し、
`gh pr merge --merge --match-head-commit <確認済みhead>` を一度だけ実行する。
各merge後も残りを再取得する。[先行mergeによるbase前進](#同じrunの先行mergeによるbase先端の前進)を実gitで検証し、
`BaseAdvance` 列と実fix diff digestをcoverage判定へ渡す。この継承条件を満たさないcoverage不整合は停止し、Codexを自動再実行しない。
base履歴を検証できないPRには `base_advance_unverified` とcoverageの変更理由、確認済みのmerge済みheadが
一致しなければ `merged_head_mismatch`（他PRには `pr_<番号>_merged_head_mismatch`）を表示する。

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
共有基盤の一致は `shared_hits`、免除時は `shared_exemption` に人間合意の根拠とコメントURLを表示する。
`auto_merge` は実効フラグ、`agent_invocations` は常に0。
終了コードは0=全件merge可/merge確認（dry-runを含む）、1=停止/未完了あり、2=入力・取得・整合エラー。
2でも成功や再レビューへ置き換えず、stderrのblocked記録を確認する。

### 共有基盤を含めた親mergeへの明示合意

人間がその親の共有基盤変更を含めて親がmergeしてよいと合意済みの場合だけ、担当親がその
発言と合意コメントURLを証拠ファイルに書き、次を実行する。既存合意の記録形式を移すための再承認は不要。

```bash
.venv/bin/python .agents/ops/review/merge_gate.py --parent 1216 shared-permission \
  --allow-shared-foundation --evidence-file tmp/shared-merge-agreement.md

# 許可の取り消しも明示記録する
.venv/bin/python .agents/ops/review/merge_gate.py --parent 1216 shared-permission \
  --deny-shared-foundation --evidence-file tmp/shared-merge-revocation.md
```

親issueにparentと同じ先頭行 `parent-session:v1` マーカーで `event=merge_permission` を記録する。
キーは `parent,event,review_run_id,request_sha256,allow_shared_foundation,evidence,revision` と
requestと同じauthorizationの3キー。旧requestでは従来のgeneration/agreement_sha256を使う。
許可フラグは真偽値、証拠は非空文章。
revisionは正整数の連番。同じ最新許可の再送は再投稿せず、取消後の再許可は新しいrevisionで記録する。
現在のgh loginによる、同じ親/run/対象digest/担当generation/実行根拠digestの最新記録だけを使う。
再合意・引継ぎ・新runで失効し、`false` の最新記録も免除しない。
この操作は共有基盤条件だけの免除であり、CI・レビュー・coverage・依存・人間判断待ちは免除しない。
`autonomous_decision` のrationale/sourcesや提案の採用チェックは、人間による共有基盤の許可ではない。
自律根拠しかない共有基盤PRはneeds_humanとなり、人間の実際の明示許可をこの別操作で記録した場合だけ免除する。

## 共有基盤パスとCI部品

`shared_paths.txt` は1行1個のrepo相対POSIX glob。`#`以降はコメント、`*`は `/` を跨がず、
`**`は跨ぐ。指定したパス自身と配下に一致し、`!`は順序によらない除外（通常パターン配下のみ）。
変更ファイルはrenameの旧パスを含め `git diff --name-only -z --no-renames` の生の名前で照合する。
`.agents/memory` は検証待ちの観察なので除外する。

`normalize_rollup` / `CheckState` は全体調査も再利用する。
workflow名＋check名ごとの最新runへ畳み、時刻なし/同時刻は最も悲観的な状態を採用する。
成功扱いはSUCCESS/NEUTRAL/SKIPPEDのみ。不明値はエラー。
`evaluate_gate(GateInput)` はCI全件・required checks・draft/fork・競合・共有パスを判定する部品で、
`review` 引数には呼出側が親run/coverage/解消証拠から確認した `ReviewOutcome` を渡す。
共有基盤に触れる場合は従来どおり人間判断が必要（明示的なキャンペーンの合意があれば親が扱う）。

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

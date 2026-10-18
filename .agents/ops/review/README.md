# 親issue単位の明示Codexレビュー

親が確定・指示したPR群を、Codexの実行1回で統合レビューする。
PR open/push/CI完了・時間経過から起動するpoll、hook、timerはない。
入力の正本は [parentのreview-request契約](../parent/README.md#1218へのレビュー要求契約schema-version-1)。
担当親自身が、同じ環境由来セッションID・ghアカウント・ローカルstateで実行する。
この入口はmerge・ラベル操作・子への修正委任を行わない。

| ファイル | 責務 |
|---|---|
| `parent_review.py` | 明示runの受理、収集、Codex実行、記録公開、親による修正確認 |
| `parent_review_policy.py` | 要求・出力・公開記録の検証とcoverageの純粋関数 |
| `review_workspace.py` | PR別headの読み取りworktree、全文diffとプロンプト |
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

`fix-evidence.json` は `ci` / `regression_tests` / `resolution` の3キーだけを持つJSON object。
すべて非空の文章とし、CIの対象head・結果へのリンク、回帰テストのコマンドと結果、指摘解消の根拠を書く。
CI・テストの実行と証拠の確認は親の責任。このコマンドはその確認を記録し、修正差分は実gitから取得する。
resolve出力のcoverageは当該PRの修正確認に関する参考判定。他PRの最新状態は取得しないため、
gateは必ず全対象の最新スナップショットを別途取得して判定する。
`--fixed-head` は現在のPR headと一致し、レビューheadの子孫でなければならない。
修正確認は親と当該PRに公開し、横断指摘は影響するPRごとに確認する。

共通の `--state-dir`（サブコマンドより前）はparent CLIと同じprivate stateを指定する。
既定値は `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/parent-sessions/`。
runの `--agent-exec` はテスト用wrapper指定、`--agent-timeout` は正の有限秒数（既定1800）。
終了コードは0=完全な結果公開または修正確認記録、1=run失敗/未完了/公開失敗、2=入力・担当・整合エラー。
`request_changes` はレビュー完了なので0だが、指摘解消まではmerge可を意味しない。

## 受理・実行・再送

1. version=1の要求を厳密に検査する。prepareだけのJSONは拒否する。
   ローカルregistryのactive担当、環境由来ID、openな親、最新の要件合意、最新run、
   同じrun ID/request digestの公開指示を照合する。
   公開指示の担当generation・agreement digestも一致させ、引継ぎ前の指示を流用しない。
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
evidenceは実行例の3キー。差分全文はローカル `resolution-<digest>.diff`、
公開済み状態は `resolution-<digest>.json` に保存する。同一証拠の再送は再投稿しない。
PRへ投稿後に親へ記録する。親に未公開のresolutionはgateへ渡さない。

## 修正後のcoverageとgate境界

`evaluate_coverage(result, current_snapshots, resolutions, changed_paths)` はI/Oのない純粋関数。
current_snapshotsは最新PR集合のスナップショット、changed_pathsは**レビューheadから最新headまでの実git差分**を渡す。
返り値は `Coverage(covered, reasons, unresolved)`。unresolvedは `(PR番号, finding ID)` の組。

- failed/incomplete・not_reviewed・PRの追加削除・base先端/base ref/merge-base/branch変更はcoverage不成立。
- head/diffが変わったPRは、同じrun/result、レビューhead、**現在のスナップショット**に対する修正確認を必要とする。
  変更パスが指摘に明記されたpathの集合を超える場合も不成立。nullの指摘から修正範囲を推測しない。
- 修正が追加pushされたら、以前のheadへの確認を流用せず、現headで各指摘の証拠を再記録する（AI再実行なし）。
- coverage成立と未解決指摘ゼロは別条件。横断指摘は影響する各PRで確認する。

未知の入力スキーマは例外で拒否する。gateは例外も停止として扱う。
最新head/CI・必須checks・競合・共有パス・人間判断・依存順の照合とmerge実行は #1219 が接続する。
この変更ではPR単位Codex承認を前提とする旧gate呼出経路を削除し、`evaluate_gate` 等の純粋部品を残す。
`AGENT_AUTO_MERGE` の設定・有効化も #1219 の担当。

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

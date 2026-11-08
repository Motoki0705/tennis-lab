# workerターンのGit保存

親の具体指示とworkerの最終報告を、実装PRとは別の追記専用ログPRへ保存する。
手動 `run-worker`、自律 `start-worker` / `continue-worker`、dispatcherの直接入口が同じ保存処理を使う。
共通runnerのcontract・result schema・終了値は変更しない。公開ログは担当registry、実行証拠、成果の受理を置き換えない。

## 保存する内容と送受信の境界

1. `worker_preflight` がtask/answer/追加指示の非空を検査し、UTF-8の原文を保持する。
   前後空白・CRLFを削らず、生成する計画判断見出しと親が書いた回答を分ける。
2. 標準指示を1回だけ読み、同じ組立て結果をrunnerへ送る。起動前にprivateなターンjournalへ
   具体指示、issue本文digest、標準指示のcommit/blob、合成promptのdigest、実行先との対応を固定する。
   合成promptを後から解析したり、元のtask.md/answer.mdを再読して原文を補ったりしない。
3. 親側がrunnerの正常終了・cleanup・runner/CLI/groupの不在とcontract・prompt digestを照合する。
   その時点のfinal.mdを読み、原文とdigestをjournalへ固定してから公開検査を行う。
   finalがあるだけ、exit 0だけでは公開しない。失敗・欠測・未終了は別の状態として残す。
4. 公開検査を通った3ファイルだけを、固定したmainから作る専用worktreeへ追加する。

stateはparent stateの `worker-logs/<repo digest>/<親>/<担当issue>/<公開キー>/<turn>/journal.json`。
新規ディレクトリ0700、journal/lockは0600。ターン単位のflock、fsyncとatomic replaceを使う。
registryロックを保持してGitHubへ公開しない。runnerディレクトリは事前作成しない。

preflightの `public_key` は私的IDとは独立したuuid4の32桁hex。
新しいlineageで作成し、既存lineageにない場合は初回captureで明示的に付与してregistryへ保存する。
自律継続でdelegation IDが変わっても同じ値を保持する。鍵・HMAC・私的IDのdigestからは生成しない。

導入前のターンにはprivateな `legacy / pre_capture_turn` を残し、公開しない。
そのturnを現在のissue本文、tmpファイル、別のrunで補わない。新しい継続turnから保存する。

## 公開パスとschema version 1

```text
.agents/logs/parents/parent-<親>/child-<担当issue>/d-<公開キー>/<4桁以上の連番>-<initial|resume>/
  instruction.md
  final.md
  meta.json
```

親番号と担当issue番号が同じ単独委任も同じ構造。共有indexはなく、確定した記録は上書きしない。
別パスへの追加と同一ターンの再送は別の契約である。パス分離だけでGitの並行更新・競合一般がなくなるとは扱わない。
`.gitignore`はログ配下を既定で除外し、parents以下のディレクトリ探索と上記3ファイルだけを許可する。
raw・stderr・README等のファイルは追跡対象にしない。

- `instruction.md`: 親が書いたtask/answerの原文を、必要な無害化後に保存する。標準指示・生成見出し・合成promptは入れない。
- `final.md`: 親側で受信を確認した最終報告の無害化後の全文。stderr・rawイベント・内部思考は対象外。
- `meta.json`: 未知キー・重複キーを拒否する。型・必須キー・参照の正本は `schema.py`。

| 群 | 内容 |
|---|---|
| `schema_version / repo / parent / issue / delegation_key` | version 1、公開repo、親・担当issue・公開系列 |
| `turn` | sequence、initial/resume、planning/implementing、previous_sequence |
| `decision` | nullable。approve/investigate/rejectと判断元の公開sequence |
| `sources` | 標準指示のcommit/path/blob、送信した担当issueのURLと本文digest |
| `instruction` | task/answer、親の追加指示原文の `additional_instructions` 配列 |
| `worker` | agent、model/max effortの選択値、観測値または明示的な未取得 |
| `runtime` | completedというターン終了の事実。タスク成功の認定ではない |
| `artifacts` | 固定した開始commit、報告に明記された同repoのPR/commit参照 |
| `files / sanitization` | 公開本文のdigest、無害化の版と置換件数 |
| `started_at / finished_at / captured_at` | timezone付き時刻。mtimeや現在時刻で終了時刻を補わない |

担当issueのdigestは送信版を識別するもので、現在のissue URLが過去本文を表示する保証ではない。
私的なID・絶対パス・合成promptのdigest・秘密の元値は公開metadataに入れない。
公開PR自身の番号はcommit後に決まるためmetaを更新せず、PR本文の固定marker・HEADとprivate journalで対応付ける。

## 公開検査

既知のcheckout絶対パスをrepo相対参照へ、Markdownのファイルリンクを固定commitのGitHub参照へ、観測済みの私的IDを明示的な伏字へ置換し、全文を再検査する。
instruction/finalだけでなく追加指示、metadataのキーと値、リンク、公開パス、branch、PR本文も対象。
既知の秘密値、UUID/session表記、資格情報の表記、未変換の絶対パス、危険なURL、制御文字を拒否する。
URLのpercent encodingも検査する。全文をhiddenにしたり、先頭だけを切り取ったりして公開成功へ変換しない。

未知の秘密情報を完全に分類する機構ではない。意味・出典を保って無害化できないものは `blocked` とし、
原文はprivateに保持する。通常出力は固定error codeだけ。Git/HTTPの診断原文もprivate journalにだけ保存する。
公開検査を省略するフラグや、任意の本文・パスを公開する入口はない。

## 専用PRと限定merge例外

専用branchは `worker-log/d-<公開キー>/<turn>`、worktreeは元repoの `.claude/worktrees/worker-log-<公開キー>-<turn>`。
実装branchへログを入れず、実装PRへログのcommitを積まない。PR本文は固定markerとRefsだけを生成し、
具体指示・最終報告や `Closes` をPR本文へ転記しない。

この入口だけは、人間との合意に基づくログ保存限定の例外として、Codexレビュー・Python check・
`AGENT_AUTO_MERGE` を要求せずにmergeする。次をすべて機械検査する。

- 期待したmainを唯一の親とする1 commit。全差分は当該turnの3ファイルの追加だけ。
- `--no-renames` の全変更種別と旧/新mode、Git treeのmode 100644/blob、期待する全バイト列。
  既存記録の改変・削除・rename元、対象外追加、symlink、gitlink、実行可能ファイルを許可しない。
- schema・本文digest・公開検査、同一repoのorigin、PRのrepo/branch/main base/固定HEAD/marker、3ファイルという件数。
- merge直前にPRを再取得し、HEAD指定でmergeする。mainが独立した変更で進んでも記録を上書きしない。
- PRがmergedであること、HEADとmerge commitがorigin/mainに含まれること、main上の保存内容を照合する。
  ここまで確認して初めて `merged / saved_to_main=true` とする。

通常の [merge gate](../review/README.md#merge-gate) とops-report memoryのCI必須処理は変更しない。
保存実装、schema、公開検査、workflow、規則を変更するコードPRは通常レビューと対象HEADのPython成功が必要。

## workflowの範囲と制約

`ci.yml`のpull_requestとmain push、labelerのpull_request_targetでは、公開領域の
instruction.md/final.md/meta.jsonだけを `paths-ignore` にする。ログのみは未起動、通常規模の混在PRは通常起動。
knowledge.ymlの対象パス・手動workflow入口は変更せず、skip ciのcommit messageで代用しない。
labelerにはpushイベントを追加しない。ログ実装のREADMEやPythonは除外対象ではない。

[GitHubのnative path filter](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#git-diff-comparisons)には差分評価上限・diff生成timeout等の制約があり、任意の巨大差分の絶対保証はしない。
本入口の3ファイルPRと通常規模の混在PRを検証対象にする。
通常・混在PRでworkflowが起動しなくても、repo内の通常merge gateは対象HEADの `Python` 成功を要求し、
欠落時にはfail-closedで止まる。未検証のコードが静かにmergeされる例外へ広げない。
mainにGitHub側の必須checkが導入された場合は、Pendingを無視せずmerge失敗を記録し、親が設定と契約を確認する。

## 状態と復旧CLI

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo owner/repo --parent 123 \
  worker-log-status --child 124
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo owner/repo --parent 123 \
  publish-worker-log --child 124 --turn 2 --dry-run
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo owner/repo --parent 123 \
  publish-worker-log --child 124 --turn 2
# 原因を照合・修復したblocked記録を、固定入力のまま再検査する
PYTHONPATH=.agents/ops .venv/bin/python -m parent --repo owner/repo --parent 123 \
  publish-worker-log --child 124 --delegation-key d-<公開キー> --turn 2 --retry
```

新しいworker/sessionを起動しない。statusは読み取り専用、dry-runはGit/HTTPの書込みもjournal更新も行わない。
公開の再開は現在の担当本人が保存済みsnapshotを照合して行う。issueのclose後も履歴を公開できるが、新しい実装を認可しない。

| 状態・error | 意味と次の対応 |
|---|---|
| `captured / awaiting_final` | 送信入力の固定だけ。元runの終了・cleanupを確認して公開CLIを再実行 |
| `ready` | finalと公開3ファイルを固定。Git保存は未完了 |
| `pending` | commit/push/PR/mergeの途中。PR作成だけでは保存成功ではない |
| `merged` | PRとorigin/mainの保存を確認済み |
| `worker_failed / missing_final` | 正常な最終報告の受信を確認できない。元runを確認し、未保存を成功へ変更しない |
| `blocked` | 公開検査、全差分、HEAD/base、dirty worktree等が不合格。固定原文を変えず原因を確認 |
| `pr_creation_uncertain` | POSTを試みたが結果不明。固定branchの全状態PRを再照合し、未確認なら再POSTしない |
| `merge_response_uncertain` | mergeを試みたが未確認。PRとmainを再照合し、盲目的に再mergeしない |
| `closed_unmerged` | 未merge close。別PRを作り直さず親へ返す |
| `legacy / pre_capture_turn` | 導入前。公開せず次の正規turnから記録 |

起動前に拒否され実行されなかった指示は `worker_failed / not_started` として残す。
`--retry` はその記録の最終報告を探さず、後続の公開可能なログを処理する。指示の再試行には別のsequenceを使う。

commit直後のjournal中断は期待した親・内容からcommitを再利用する。PR作成直後・merge直後の応答喪失も
同じbranch・marker・HEADを照合して再利用する。dirtyな専用worktreeや破損journalを自動reset・初期化しない。
親が対象worktreeだけを明示修復してから再検査する。非公開原文、固定payload、HEADを書き換えて復旧を装わない。

自動hookのworker終了値は従来どおりrunnerの終了値で、手動応答の `log_publication` とこのCLIの状態が保存結果。
公開CLIはエラーがあれば終了値2。保留の0も保存成功を意味せず、`saved_to_main` を確認する。
定期poll、新しいtimer、追加agent、通常gateの変更は導入しない。

## 検証

unitは公開schema・無害化・追加専用policy、e2eはfake GitHubと使い捨て実gitで公開、並行再送、障害復旧を検証する。
既存の手動・自律planningテストを拡張し、実際の送信原文、回答、追加指示、phase/modelの対応も確認する。
通常gateのPython必須は既存 `test_python_ci_is_required_for_dependency_ordered_merges` を維持する。

実運用の受入確認はコードPRの通常merge後に親が指示する。それまでは未確認とする。

1. 公開可能な正規turnを専用CLIで公開・mergeし、journal・3ファイル・PR・mainを照合する（来歴、冪等性、保存成功）。
2. CIとlabelerのrun履歴をイベント・PR番号・merge commitで取得し、ログPRとmain mergeで未起動を確認する。
   pull_request_targetはdefault branchのworkflowを使うため、PR headのcheck一覧だけで判定しない（CI省略）。
3. 別の通常/混在probe PRで従来のworkflow起動を確認する。混在probeは実装PRへ入れず、確認後に未mergeで閉じる。
4. 対照PRのCI完了後にも履歴を再取得し、照会時刻・対象workflow・PR番号・commitを実装PRへ記録する。

# 親の一致判定用資料

workerが各モデルの主張を要約し、対応しそうな正解IDを挙げた資料。対応候補は一致の判定ではない。同じpathや同じfinding IDでも同一論点とは限らず、pathが異なっても同じ障害を指す場合がある。正誤・検出・見逃し・誤指摘・valid_newの判断は全て親が行う。

発生条件・処理・影響・修正対象はモデルの主張の要約であり、新たに再現・検証した結果ではない。正解自体の妥当性確認は既存のgold.jsonのテスト・再現・親確認へリンクする。

## pr-1699

### 正解論点 `pr-1699-1`: URLの資格情報検査をスキームの大小文字で回避できないこと

正解severity: major。場所: `.agents/ops/worker_logs/safety.py:21`。

[既存の妥当性確認](https://github.com/Motoki0705/tennis-lab/pull/1699#issuecomment-6700364417): 親が再実行した結果: tests/unit/agents_ops、tests/e2e/agents_ops/test_worker_log_publication.py、tests/e2e/development/test_ci_test_lanes.py で 767 passed。 今回この既往テストは再実行していない。

- Astra `pr-1699-1` ([出力](../../../knowledge/runs/run-i1863-pr1699-astra-20261111/output.json)): 大文字のHTTPSスキームとuserinfoを持つURLが、既知private値に含まれない場合。
  処理: URL抽出はcase-sensitiveだが、絶対パス検査はcase-insensitiveでHTTP(S)を除外する、というモデルの指摘。
  影響: 指示やfinal内の資格情報付きURLが拒否されずGit公開される、という主張。
  修正対象: `.agents/ops/worker_logs/safety.py:21`。親判定: pending。

- Sol `pr-1699-1` ([出力](../../../knowledge/runs/run-i1863-pr1699-sol-20261111/output.json)): 大文字HTTPSとuserinfoのURLが既知private値に含まれない場合。
  処理: URL正規表現と絶対パス検査で大小文字の扱いが異なる、というモデルの指摘。
  影響: 認証情報付きURLがinstruction/finalから公開・自動mergeへ通る、という主張。
  修正対象: `.agents/ops/worker_logs/safety.py:21`。親判定: pending。

### 正解論点 `pr-1699-2`: 追跡例外を公開対象のログファイルへ限定すること

正解severity: major。場所: `.gitignore:87`。

[既存の妥当性確認](https://github.com/Motoki0705/tennis-lab/pull/1699#issuecomment-6700368731): 親が再実行した結果: tests/unit/agents_ops、tests/e2e/agents_ops/test_worker_log_publication.py、tests/e2e/development/test_ci_test_lanes.py で 767 passed。 今回この既往テストは再実行していない。

- Astra: workerは対応候補を抽出していない。親が全指摘を確認して判定する。

- Sol: workerは対応候補を抽出していない。親が全指摘を確認して判定する。

### 正解表への対応をworkerが割り当てていない指摘

- Astra `pr-1699-2` ([出力](../../../knowledge/runs/run-i1863-pr1699-astra-20261111/output.json)): dispatcherがrepo名を小文字化し、GitHubが正規表記のPR URLを返す場合。
  処理: repo比較はcasefoldだがhtml_urlは入力repoとの完全一致を要求する、というモデルの指摘。
  影響: 正当なログPRがpr_identity_changedで拒否され、自動mergeまで進まない、という主張。
  修正対象: `.agents/ops/worker_logs/schema.py:235`。親判定: pending。

- Astra `pr-1699-3` ([出力](../../../knowledge/runs/run-i1863-pr1699-astra-20261111/output.json)): PUT送信前の記録後に通信障害やAPI拒否が起き、PRが未merge・openのままの場合。
  処理: merge_attemptedが永続化され、その後の再送をmerge_response_uncertainで拒否する、というモデルの指摘。
  影響: 障害解消後も同じログPRをmainへ保存できない、という主張。
  修正対象: `.agents/ops/worker_logs/publication.py:435`。親判定: pending。

## pr-1766

### 正解論点 `pr-1766-1`: 自動継続はtimed_outに限定し、取消は明示retryまで保持する

正解severity: major。場所: `src/tasks/annotation/recovery.py:80`。

[既存の妥当性確認](https://github.com/Motoki0705/tennis-lab/pull/1766#issuecomment-6727917767): 親がローカルで .venv/bin/python -m pytest -n 0 tests/unit/tasks/annotation/test_local_controller.py::test_cancellation_after_automatic_decision_requires_explicit_retry を実行し、1 passed（4.91秒）。workerは修正前のコードでこのテストが失敗することを再現したと報告している。test_workers.py に accepted_states の検証テストも追加されている。 今回この既往テストは再実行していない。

- Astra `pr-1766-1` ([出力](../../../knowledge/runs/run-i1863-pr1766-astra-20261111/output.json)): 継続判断の保存とauthorize_retryの間に、同じattemptのstopがcancelledを確定する場合。
  処理: work lock解放後のauthorize_retryがcancelledも受理しreadyへ上書きする、というモデルの指摘。
  影響: 取消済みcameraが次tickで自動再起動する、という主張。
  修正対象: `src/tasks/annotation/recovery.py:80`。親判定: pending。

- Sol `pr-1766-1` ([出力](../../../knowledge/runs/run-i1863-pr1766-sol-20261111/output.json)): timeoutと競合するpool.stopが、判断保存後のロック解放中に同じattemptをcancelledへ確定する場合。
  処理: authorize_retryがattempt一致だけでcancelledをreadyへ戻す、というモデルの指摘。
  影響: 明示retry待ちの要件に反し、取消済みcameraが自動起動する、という主張。
  修正対象: `src/tasks/annotation/recovery.py:80`。親判定: pending。

### 正解表への対応をworkerが割り当てていない指摘

- Astra `pr-1766-2` ([出力](../../../knowledge/runs/run-i1863-pr1766-astra-20261111/output.json)): 全frame保存後のtimeoutで欠損・破損によりholdを保存し、その後ファイルを修復した場合。
  処理: cycleが不変holdを再利用し、RetryWorkerも全frame処理済みを理由に拒否する、というモデルの指摘。
  影響: 保存修復後も正式recover APIで完成receiptを確定できない、という主張。
  修正対象: `src/tasks/annotation/recovery.py:410`。親判定: pending。

## pr-1867

### 正解論点 `pr-1867-1`: 古い採用元で後続runの未解決指摘を隠さない

正解severity: major。場所: `.agents/ops/review/parent_review.py:769`。

[既存の妥当性確認](https://github.com/Motoki0705/tennis-lab/pull/1867#issuecomment-6767712600): 担当親がworktreeで `-n 0` 実行: tests/e2e/agents_ops/test_review_runtime_evidence.py と test_parent_review_policy.py で111 passed（6.16秒）、test_review_recovery.py の -k "later_findings or inconsistent_runner or event_final or laundered" で6 passed（39.04秒）。workerは関連8ファイル397ケースの `-n 0` 成功と、指摘ごとの故障注入2件の検出を報告（PR #1867 本文）。 今回この既往テストは再実行していない。

- Astra `pr-1867-1` ([出力](../../../knowledge/runs/run-i1863-pr1867-astra-20261111/output.json)): 同一snapshotのAにapprove finalがあり、後続Bが重大指摘を公開した後にCでAを採用する場合。
  処理: recoverは元runとsnapshotだけを確認し、gateは最新Cを選ぶ、というモデルの指摘。
  影響: Bの指摘をresolveせず親確認とCIだけでgateを通せる、という主張。
  修正対象: `.agents/ops/review/parent_review.py:769`。親判定: pending。

- Sol `pr-1867-1` ([出力](../../../knowledge/runs/run-i1863-pr1867-sol-20261111/output.json)): 同一snapshotのR1にapprove finalがあり、後続R2がmajorを報告した後にR3がR1をrecoverする場合。
  処理: gateは最新R3を選び、verify_adoptionはR1との整合だけで後続R2を照合しない、というモデルの指摘。
  影響: R2の指摘をresolveせず親確認とCI成功があればgateを通れる、という主張。
  修正対象: `.agents/ops/review/merge_gate.py:292`。親判定: pending。

### 正解論点 `pr-1867-2`: 実行証拠の内容不一致を復旧で承認へ変換しない

正解severity: major。場所: `.agents/ops/review/review_recovery.py:50`。

[既存の妥当性確認](https://github.com/Motoki0705/tennis-lab/pull/1867#issuecomment-6767715093): 担当親がworktreeで `-n 0` 実行: tests/e2e/agents_ops/test_review_runtime_evidence.py と test_parent_review_policy.py で111 passed（6.16秒）、test_review_recovery.py の -k "later_findings or inconsistent_runner or event_final or laundered" で6 passed（39.04秒）。workerは関連8ファイル397ケースの `-n 0` 成功と、指摘ごとの故障注入2件の検出を報告（PR #1867 本文）。 今回この既往テストは再実行していない。

- Astra: workerは対応候補を抽出していない。親が全指摘を確認して判定する。

- Sol: workerは対応候補を抽出していない。親が全指摘を確認して判定する。

### 正解表への対応をworkerが割り当てていない指摘

- Astra `pr-1867-2` ([出力](../../../knowledge/runs/run-i1863-pr1867-astra-20261111/output.json)): 非ゼロ終了runの検証済み候補を保存後、失敗結果の公開中に本体が停止しexecutionがrunningのままの場合。
  処理: recoverはrunningを拒否し、resume-publicationはoutputと失敗resultの一致要求で拒否する、というモデルの指摘。
  影響: 終了とcleanupを確認できてもAI起動0回の復旧経路が無くなる、という主張。
  修正対象: `.agents/ops/review/review_recovery.py:109`。親判定: pending。

## pr-1872

### 正解論点 `pr-1872-1`: 終了・cleanup未確認のレビューは公開状態にかかわらず同PRの起動を止める

正解severity: major。場所: `.agents/ops/review/review_locks.py:78`。

[既存の妥当性確認](https://github.com/Motoki0705/tennis-lab/pull/1872#issuecomment-6773515966): 親がPR headで `pytest -n 0 tests/e2e/agents_ops/test_review_start_boundary.py tests/e2e/agents_ops/test_review_interruption.py tests/e2e/agents_ops/test_review_concurrency.py` を実行し、27件が成功（151.93秒）。workerは関連99ケースの成功と、各指摘の新しい検査を外す故障注入2件の検出を報告している。 今回この既往テストは再実行していない。

- Astra `pr-1872-1` ([出力](../../../knowledge/runs/run-i1863-pr1872-astra-20261111/output.json)): agent-runtimeがSIGKILLされ、別sessionで動くCLIが残っているのに失敗結果の公開まで進んだ場合。
  処理: 排他検査はexecution.statusのrunningだけを拒否し、終了・cleanup証拠を見ない、というモデルの指摘。
  影響: cleanup未確認の旧CLIと同じPRの新runが並行起動できる、という主張。
  修正対象: `.agents/ops/review/review_locks.py:78`。親判定: pending。

- Sol `pr-1872-1` ([出力](../../../knowledge/runs/run-i1863-pr1872-sol-20261111/output.json)): 共通runnerだけが強制終了して旧CLIが残り、agent工程unconfirmedでも公開処理でexecutionがpublishedまたはpublication_failedになった場合。
  処理: running以外を通す排他検査が、起動済みruntimeの終了とcleanupを確認しない、というモデルの指摘。
  影響: lease解放後に同PRの別runが新agentを並行起動できる、という主張。
  修正対象: `.agents/ops/review/review_locks.py:78`。親判定: pending。

### 正解論点 `pr-1872-2`: runtime記録作成前の中断にも検証付き終結経路を用意する

正解severity: major。場所: `.agents/ops/review/review_interruption.py:98`。

[既存の妥当性確認](https://github.com/Motoki0705/tennis-lab/pull/1872#issuecomment-6773519361): 親がPR headで `pytest -n 0 tests/e2e/agents_ops/test_review_start_boundary.py tests/e2e/agents_ops/test_review_interruption.py tests/e2e/agents_ops/test_review_concurrency.py` を実行し、27件が成功（151.93秒）。workerは関連99ケースの成功と、各指摘の新しい検査を外す故障注入2件の検出を報告している。 今回この既往テストは再実行していない。

- Astra: workerは対応候補を抽出していない。親が全指摘を確認して判定する。

- Sol: workerは対応候補を抽出していない。親が全指摘を確認して判定する。

### 正解表への対応をworkerが割り当てていない指摘

- Astra `pr-1872-2` ([出力](../../../knowledge/runs/run-i1863-pr1872-astra-20261111/output.json)): AI未起動のrecover adoptionがrunningを保存して公開前に中断し、後でhead/baseが進んだ場合。
  処理: close-interruptedが未起動runを起動済みと扱い、作成されていないruntime resultを必須にする、というモデルの指摘。
  影響: snapshot更新後はrecover再送と公開再開もできず、同PRの新runを恒久的に止める、という主張。
  修正対象: `.agents/ops/review/review_interruption.py:23`。親判定: pending。

- Sol `pr-1872-2` ([出力](../../../knowledge/runs/run-i1863-pr1872-sol-20261111/output.json)): runtimeを作らないrecover adoptionが公開途中で中断し、その間にhead/baseが進む場合。
  処理: not_invokedも起動済み扱いになり、close-interruptedが存在しないruntimeを要求する、というモデルの指摘。
  影響: snapshot更新で再送と公開再開も拒否され、残存runningが同PRの新runを阻止する、という主張。
  修正対象: `.agents/ops/review/review_interruption.py:23`。親判定: pending。

## pr-1888

### 正解論点 `pr-1888-1`: PR単位で採用可能な結果にも実効モデル検証を適用する

正解severity: major。場所: `.agents/ops/review/review_recovery.py:347`。

[既存の妥当性確認](https://github.com/Motoki0705/tennis-lab/pull/1888#issuecomment-6785950914): 親がPR headで `pytest -n 0 tests/unit/agents_ops/test_review_settings.py tests/e2e/agents_ops/test_review_choice_handoff.py tests/e2e/agents_ops/test_review_model_selection.py` を実行し、60件が成功した（207.28秒）。workerは追加15ケース（unit 7、E2E 8）の成功と、故障注入3件の検出を報告している。 今回この既往テストは再実行していない。

- Astra `pr-1888-1` ([出力](../../../knowledge/runs/run-i1863-pr1888-astra-20261111/output.json)): あるPRがreviewed、別PRがnot_reviewedとなり、run全体がfailedまたはincompleteの場合。
  処理: 全体の成功状態だけでruntime検証を省略する一方、coverageとgateはreviewedのPRを独立採用する、というモデルの指摘。
  影響: 保存後の実効設定の不一致や証拠欠損を照合せず、部分結果のPRを承認へ採用できる、という主張。
  修正対象: `.agents/ops/review/review_recovery.py:347`。親判定: pending。

- Sol `pr-1888-1` ([出力](../../../knowledge/runs/run-i1863-pr1888-sol-20261111/output.json)): PR Aがreviewed、PR Bがnot_reviewedで、全体がfailedまたはincompleteになる場合。
  処理: 早期returnがruntime検査を省略する一方、coverageとgateはAを採用できる、というモデルの指摘。
  影響: runtime記録の欠落やモデル・effortの不一致、digestや公開設定の不整合を検出せず、公開再開とAのmergeを許す、という主張。
  修正対象: `.agents/ops/review/review_recovery.py:347`。親判定: pending。

### 正解論点 `pr-1888-2`: handoffで保存済み候補の指摘を脱落させない

正解severity: major。場所: `.agents/ops/parent/review_choice.py:59`。

[既存の妥当性確認](https://github.com/Motoki0705/tennis-lab/pull/1888#issuecomment-6785954205): 親がPR headで `pytest -n 0 tests/unit/agents_ops/test_review_settings.py tests/e2e/agents_ops/test_review_choice_handoff.py tests/e2e/agents_ops/test_review_model_selection.py` を実行し、60件が成功した（207.28秒）。workerは追加15ケース（unit 7、E2E 8）の成功と、故障注入3件の検出を報告している。 今回この既往テストは再実行していない。

- Astra: workerは対応候補を抽出していない。親が全指摘を確認して判定する。

- Sol: workerは対応候補を抽出していない。親が全指摘を確認して判定する。

### 正解表への対応をworkerが割り当てていない指摘

- Astra `pr-1888-2` ([出力](../../../knowledge/runs/run-i1863-pr1888-astra-20261111/output.json)): human_instruction指定でreview-modelをAstra、human-review-effortをmaxとし、human-review-modelを省略して理由と参照を渡した場合。
  処理: human modelが無いためhuman effortを通常effort引数で上書きし、明示maxが消える、というモデルの指摘。
  影響: 拒否すべきeffortを既定xhighに変え、人間指定として記録・実行できる、という主張。
  修正対象: `.agents/ops/lib/review_settings.py:210`。親判定: pending。

## pr-1686

正解表は暫定負例。独立した静的確認の範囲・限界はgold.jsonのnegative_reviewを参照。モデルの指摘があれば、正解表外として親が個別に確認する。

## pr-1744

正解表は暫定負例。独立した静的確認の範囲・限界はgold.jsonのnegative_reviewを参照。モデルの指摘があれば、正解表外として親が個別に確認する。

## 判定を返す形式

`adjudications.json`のrun_id、output_sha256、gold_sha256を維持し、親がadjudicatorをparentへ変更する。matchesはfinding_id、gold_id、reason、evidence、other_findingsはfinding_id、classification（false・unconfirmed・valid_new）、reason、evidenceを記入する。公開projectionの指紋を元出力の指紋へ上書きしない。記入後、非公開stateに対するreportのparent-adjudicationsで数値を再計算する。未記入templateは判定結果として受理されない。

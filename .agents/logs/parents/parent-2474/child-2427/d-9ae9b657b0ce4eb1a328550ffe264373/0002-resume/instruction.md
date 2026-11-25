PR #2481 の正式レビュー（run parent-2474-bfe274cd6367424db84f83c7cdc1ba6d, round 1）で major が4件出た。すべて採用するので、同じbranch・PRで修正する。各指摘の原文を下に示す。

## 指摘1: HEAD更新で自動レビュー進行が終了しても親に通知されない

実行検証（GitHub・HEAD・状態保存をスタブ化）。PRのマージ後にmain CIがpendingの状態でHEAD更新を発生させると、`checkout_wait()` の終了値75がcontrollerの例外処理を通過し、保存状態は `running`、対象は `post_merge_wait`、通知は0件のままでした。controllerのstderrは `controller.log` に入り、更新後の `wait-merge` も既定では新しいイベントを無期限に待ちます。版不一致で旧コードを止める契約を保ちつつ、親が中断を観測し、同じrunから再開できるようにしてください。既存のCI再開テストは同一HEADでの `KeyboardInterrupt` を扱うため、HEAD変更時の通知と再開を検証できません。[進行ループ](https://github.com/Motoki0705/tennis-lab/blob/fd1d8e6bba067518e689fede7092b7466af49825/.agents/ops/review/progression.py#L387) [controller起動と待機](https://github.com/Motoki0705/tennis-lab/blob/fd1d8e6bba067518e689fede7092b7466af49825/.agents/ops/review/merge_gate_impl.py#L468)

## 指摘2: メインから直接起動した監視は75で終了すると同じ待機を再開できない

実行検証（registry・HEAD・ジョブ観測をスタブ化）。READMEに記載されたメインcheckoutからの直接起動で、監視がclaimを保存した後、ジョブ待機中にHEADを更新しました。監視は75で終了し、記録は `waiting` かつ `monitor_started=true` のままでした。同じ引数で起動し直すと `claim()` が拒否し、監視を始めずに0を返します。`wait-worker` からは `followup_controller_unconfirmed` になります。直接起動でも安全に再開できるようにするか、claim前に固定worktreeでの起動を要求してください。既存の監視テストは固定worktreeのサービス経路を検証しています。[監視のclaimと復帰処理](https://github.com/Motoki0705/tennis-lab/blob/fd1d8e6bba067518e689fede7092b7466af49825/.agents/ops/parent/worker_monitor.py#L216)

## 指摘3: 手動ワーカーの自動修正中はレビュー進行が上限なしでcheckout lockを保持する

静的な推論。`merge_gate.py` の手動ワーカー向け修正は `run_worker()` を同期実行し、`execute_manual()` は `communicate()` で終了を待ちます。runnerの起動引数に `--timeout` がなく、controllerとメイン由来のrunnerがターン終了まで共有lockを保持するため、修正が長引くとgit-syncは引き続き見送られます。READMEの「実行と回収が終わるまで」には実効上限がありません。同じ表の `explain_html serve` も、リクエストを待つ間を含め停止まで保持します。受入条件2に沿って、これらの入口で待機中にlockを解放するか、保持する理由と実効上限を明記し、その契約を検証してください。[手動修正経路](https://github.com/Motoki0705/tennis-lab/blob/fd1d8e6bba067518e689fede7092b7466af49825/.agents/ops/review/progression_backend.py#L358) [同期runner待機](https://github.com/Motoki0705/tennis-lab/blob/fd1d8e6bba067518e689fede7092b7466af49825/.agents/ops/parent/improvement_runtime.py#L166) [ロックの一覧](https://github.com/Motoki0705/tennis-lab/blob/fd1d8e6bba067518e689fede7092b7466af49825/.agents/ops/lib/README.md#L142)

## 指摘4: 通知処理の開始前に中断すると連続見送りの通知が抑止される

実行検証（状態ファイルと通知先をスタブ化し、SIGTERM相当の例外を注入）。3回目の見送りで `uncertain` を保存した直後、`consecutive_skips` の記録直前に中断すると、4回目も `uncertain` を引き継ぎ、しきい値到達のjournal記録もWindows通知の試行も0件でした。送信を試みた可能性のある状態で重複を抑える方針は維持できます。しきい値到達を先に記録し、送信前と送信結果不明を区別できるようにしてください。既存の再試行テストは送信関数が呼ばれた後の例外のみを扱っています。[通知状態の保存順序](https://github.com/Motoki0705/tennis-lab/blob/fd1d8e6bba067518e689fede7092b7466af49825/.agents/ops/git_sync/cli.py#L134)

## 求める修正

- 指摘1・2: HEADの更新で旧コードを止める契約（終了値75）は保つ。そのうえで、review進行のcontrollerとworker_monitorの直接起動のどちらでも、中断を親が観測でき（状態・通知に残る）、同じrun・同じ待機を安全に再開できるようにする。固定worktreeでの起動を要求する方式を選ぶなら、claimの前に拒否して状態を残さないこと。HEADが変わったときの通知と再開を回帰テストで確かめる。
- 指摘3: 手動ワーカーの自動修正中の同期runner待機と  について、待機中にlockを解放するか、保持する理由と実効上限を README の表に書き、その契約をテストで示す（上限なしの保持を残さない）。
- 指摘4: しきい値到達を先に記録し、「送信前」と「送信結果不明」を区別する。中断の位置によって通知が抑止されないことを回帰テストで確かめる。
- 同じ親の PR #2477（#2392）が先にマージされる見込みで、 の同じ箇所で競合する。push前に最新mainを取り込み、両方の記述を残して競合を解消する。
- 検証とpre-commitを通してcommit・pushし、PR番号と完全なHEADを宣言して終了する。

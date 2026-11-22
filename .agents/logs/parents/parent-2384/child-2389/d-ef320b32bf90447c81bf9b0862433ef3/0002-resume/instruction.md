親Issue #2384 配下の子Issue #2389（PR #2398、run `parent-2384-a4ce3281d6fd4f6fb2f7322299258727`）の正式レビューで検出された major 指摘 4 件をすべて修正し、検証・commit・push まで行ってください。

## 修正対象の major 指摘 4 件（すべて採用・修正必須）

### Major 1: 環境不備を拒否する前にログ捕捉と公開続行が走る
- 指摘詳細:
  `cli_impl.py:678`・`improvement_runtime.py:127` で、`.venv/bin/python` が最初から欠けていても、環境検査による拒否より前に既存の公開待ちログの公開続行と判断ログ捕捉が走り、さらに Popen 直前の再検査も worker ログ捕捉と予約保存の後に走っているため、環境不備時に副作用（判断ログ書き込み・公開進行・取消済みターンとログ残骸）が生じる。
- 修正要件:
  - `run-worker` のランナー実行環境検査を、CLI の公開続行・判断ログ捕捉・予約保存・worker ログ捕捉のいずれよりも**先**に実行し、最初から不備がある場合は一切の副作用なしに理由付きで拒否する。
  - Popen 直前再検査で初めて不備が見つかった場合も含め、不要な副作用を残さないことを確認する。
  - 公開待ちログ・判断ログの書き込みを監視し、環境不備での `run-worker` 拒否時に公開続行も判断ログ・予約・worker ログ書き込みも起きない回帰テストを追加する。

### Major 2: 取消済みの未起動ログが親の公開残件に残り続ける
- 指摘詳細:
  `reconciliation.py:116`・`service.py:265` で、`cancel-worker-launch` 後の `worker_failed / not_started / runner_not_started` の journal を `remaining_publications()` が公開残件として返してしまう。公開 CLI は `not_started` を処理対象外にするため、同じ委任を再起動して後続ターンを公開しても親の `status` / `release --ended` の公開残件がゼロにならず完了できない。
- 修正要件:
  - `not_started`（未起動終結済み）の journal は `remaining_publications()` および終了チェックポイントの未公開残件に含めないようにする（ただし不正・改ざん・未確認の journal は従来どおり残件・エラーとして扱う）。
  - `cancel-worker-launch` → 同一委任で再起動・後続ターン公開の E2E で、`remaining_publications()` および `status` / 終了チェックポイントの公開残件がゼロになる回帰テストを追加する。

### Major 3: 後の手動起動によって同じ待機指定が別のrunを返す
- 指摘詳細:
  `worker_wait.py:116`・`manual_workers.py:858` で、最初の手動 run A が自動再開 B まで完了した状態では `--after-run none` が B を選ぶのに、同じ委任で次の手動 run C を記録すると同じ指定が A を選んでしまう（手動起動が `manual_followup` を上書きするため元の自動再開先を辿れない）。また、A の後の手動起動を取り消すと取消側の `run_dir` が A に戻り、A に別起動のエラーが結び付いてしまう。
- 修正要件:
  - 自動再開先や各起動系列の対応付けが、後続の手動起動や別起動の取消によって上書き・混線しないようにする（例: 各 run/followup の系列履歴を保持し、取消時に別 run へエラーを付け戻さない）。
  - A → 自動再開 B → 後続手動 run C の後でも `--after-run none` の解決が一貫すること、および A の後の手動起動を取り消しても A に取消エラーが結び付かないことの回帰テストを追加する。

### Major 4: 正式レビューの手動修正ターンで起動待ちと未起動復旧が成立しない
- 指摘詳細:
  `manual_workers.py:933`・`worker_wait.py:261` で、直前 run の後に修正用（`repair`）予約だけが保存された状態では対応する `manual_followup` がないため、`--after-run` が runner 記録の出現を待たず `runner_records_unavailable_or_invalid`（終了値 2）で即終了する。また launcher 終了・run dir 不在の修正ターンを `cancel-worker-launch` に渡すと `repair` を理由に拒否され、修正予約が `starting` に残ったまま再起動不能になる。
- 修正要件:
  - 修正IDの二重実行を厳格に防ぎつつ、修正ターンでも `--after-run` が予約保存から runner 記録の出現・終了まで待てるようにする。
  - また launcher 終了・run dir 不在（AI 未起動確認済み）の修正ターンについて、`cancel-worker-launch` で安全に未起動予約を取り消し（`launch_state` を適切に戻し／二重実行を防ぎつつ）再試行できるようにする。
  - 正式レビューの手動修正ターンにおける `--after-run` 起動待ちおよび未起動取消・再試行の回帰テストを追加する。

## 検証・push・終了ルール
- 関連テスト（新規回帰テスト＋既存の manual_workers / worker_wait / worker_followup / worker_logs テスト）を実行し全通過を確認する。
- pre-commit（ruff・mypy）を通して commit・push し、PR #2398 本文（`Closes #2389` を維持）に major 4件ごとの修正 commit と検証結果を追記する。
- push が終わったら、このターン内で CI 待ちループを行わず、最終メッセージ末尾に以下の形式で 1 行だけ出力して終了する（`<40桁HEAD SHA>` は push した実際の値に置換）：
`PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2398,"head":"<40桁HEAD SHA>","resume_on_success":false}}`

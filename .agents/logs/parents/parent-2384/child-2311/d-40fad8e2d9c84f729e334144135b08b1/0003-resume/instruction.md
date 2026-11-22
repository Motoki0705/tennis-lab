親Issue #2384 配下の子Issue #2311（PR #2400、run `parent-2384-fb17100d929d4044ac66ee45bdebe416`）の正式レビューで検出された major 指摘 3 件をすべて修正し、検証・commit・push まで行ってください。

## 修正対象の major 指摘 3 件（すべて採用・修正必須）

### Major 1: 証拠Issueの作成が明示拒否されると担当開始記録を回収できない
- 指摘詳細:
  `evidence.py:192-200` は Issue 作成 POST の前に番号なしの `sent` を保存する。POST が HTTP 403 等で明示拒否されてもこの `sent` 記録が応答不明と同じ扱いで残るため、権限を修復して `assignment_evidence.retry` の案内どおり `register` や `evidence --create` を再実行しても、一覧に作成済み Issue がないため `find()` が拒否し続け、担当保存済みのまま `assigned` を回収できない。
- 修正要件:
  - 証拠 Issue 作成の POST でも、HTTP 4xx 等の**明示拒否**（作成されていないことが確定した応答）と通信断・timeout・5xx 等の**応答不明**を区別し、明示拒否の場合は番号なし `sent` を残さずロールバック（または明示拒否として解消後に再作成可能）にする。
  - 応答不明時は従来どおり重複 POST を防ぐ安全契約を維持する。
  - 証拠 Issue 作成自体が HTTP 403 で拒否された後に、権限回復→`register` 再実行で証拠 Issue と `assigned` が各 1 件だけ作成・回収される E2E 回帰テストを追加する。

### Major 2: 担当世代の開始経路によって記録欠落を見逃す
- 指摘詳細:
  `session.py:270` 付近で、dispatcher 由来の担当を手動で引き継ぎ、`release` 後に同じセッションで `register` すると、新世代に旧 `dispatch` が残り `handoff_generation` は消える。この世代の `assigned` 投稿が 403 で拒否された場合、`_missing_assignment_record()` が `dispatch` の存在だけで検査を省略するため、`status` に欠落が出ず `agree` や `create-child` に進めてしまう。また、`handoff --accept` の記録が欠落した現世代で `register` を実行すると、必要な `handoff` の代わりに `assigned` を書いて成功を返し、実行時の欠落判定が残り続ける。
- 修正要件:
  - 現世代の担当開始経路（手動 `register` / 手動 `handoff --accept` / dispatcher 起動）を明示的に記録・判定し（`register` による新世代開始時に旧 `dispatch` 等の前世代属性を引きずらない）、現世代に必要な開始記録種別（`assigned` か `handoff` か）を一意に決める。
  - `handoff --accept` の記録欠落状態で `register` を呼ばれた場合に誤って `assigned` を書かず、正規の `handoff --accept` 再実行を案内または拒否する（両経路で誤った記録種別を書き込まない）。
  - 上記 2 シナリオ（dispatcher 由来→引継ぎ→`release`→`register` で 403 拒否後の欠落検出・実行拒否・再実行 1 件回収、および `handoff --accept` 記録欠落時に `register` が誤記録せず `handoff --accept` 再実行で 1 件回収されること）の回帰テストを追加する。

### Major 3: 停止節の一括除外で要件の追記を見逃し、正規の停止入力でも承認が壊れる
- 指摘詳細:
  `improvement.py:148` の `contract()` が見出し名だけで「作業状況・停止事項」全体を除外しているため、停止記録が存在しない節でも承認後に追記された `###` の追加要件や自由文を照合せず `create-child` を通してしまう。一方、`session.py:1643` の `stop` は複数行の `facts` 等をそのまま本文へ書くため、そこに含まれる `##` 見出し行が独立した未知節になり、正規の停止操作直後でも `requirements changed` で分割できなくなる。
- 修正要件:
  - `contract()` での除外対象を「見出し名による『作業状況・停止事項』全体の一括除外」ではなく、**CLI の `stop` / `resume` が記録した構造化停止・再開記録（`parent-task-status:v1` を伴う CLI 管理範囲）のみ**に厳密に限定する（停止節内に手書き追記された要件や `###` 追加要件は承認照合の対象として検知・拒否する）。
  - `session.py` の `stop` / `resume` で複数行の入力（`facts`, `reason` 等）に `## ` 見出し行等が含まれていても、Markdown のセクション境界を壊して独立した `##` 節にならないよう安全に書式化（例: 構造化表現・インデント／引用化・または入力検証）する。
  - 停止節内に `###` 追加要件や自由文を追記すると `contract()` 照合で拒否されるテストと、`##` 行を含む `facts` 等で正規の `stop --child <親番号>` を行っても `create-child` が通り承認が壊れないテストの双方を追加する。

## 検証・push・終了ルール
- 関連テスト（新規回帰テスト＋既存の evidence / parent_session / improvements / herdr 関連テスト）を実行し全通過を確認する。
- pre-commit（ruff・mypy）を通して commit・push し、PR #2400 本文（`Closes #2311` を維持）に major 3件ごとの修正 commit と検証結果を追記する。
- push が終わったら、このターン内で CI 待ちループを行わず、最終メッセージ末尾に以下の形式で 1 行だけ出力して終了する（`<40桁HEAD SHA>` は push した実際の値に置換）：
`PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2400,"head":"<40桁HEAD SHA>","resume_on_success":false}}`

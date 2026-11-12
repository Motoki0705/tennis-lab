# PR #1927 再レビュー指摘の修正（run parent-1881-91e9b778f10e47788a245dabbf8a51be、HEAD 73242cc4、request_changes）

前回の2件は修正済みと確認された。#1883 統合に伴う major 3件を、親が妥当と判断した。同じ branch・PR で修正する。

## pr-1927-1（`.agents/ops/dispatcher/schema.py:82`）
`source_url` が新しい note 等の引用を証拠用Issueの実URLへ変換し、`supervisor._validate_start`（379行）と `_job`（582行）がそのURLの証拠用Issue番号を collect の作業対象へ直接渡す。collect 内の `gh.issue` が「parent evidence issue cannot be used as a work issue」で拒否し、正常な証拠コメントを引用した startable 候補の起動予約が取り消され、継続予約も失敗する。
- judgement 側と同様に、保存済み snapshot から `history_owner` で所属Issueへ解決する。
- 回帰テスト: 証拠用Issueのコメントを引用した候補が、supervisor の起動・継続予約まで通ること（既存テストは旧親コメントかIssue自体の引用だけ）。

## pr-1927-2（`.agents/ops/decision_logs/parent_commands.py:603`）
`improvements.append` は候補を証拠用Issueへ保存するが、`reason_reference` の improvement 分岐は親Issue番号でURLを組み立てる。存在しない親コメントへの参照が公開証拠として保存され、理由の正本を辿れない。
- 追加済みの `improvements.comment_url` をこの分岐でも使う。
- 回帰テスト: 切り出し decision 側の reason_ref URL が実在する証拠用Issueのコメントを指すこと。

## pr-1927-3（`.agents/ops/parent/notifications.py:281`）
`progress_lines` は過去に `workflow_completed` が1件でもあれば「現在の段階: workflow完了照合済み」を生成する。完了後に受入条件を追加して再承認・agree する経路（#1883）では `completion.status` は unverified になるのに、agree 後の自動 sync が本文へ完了表示と旧通知の「未達・残件: なし」を書き戻す。
- 現在の受入の適用可否を共通の completion 判定から取得し、過去の成果と現在の未確認状態を区別して表示する。
- 回帰テスト: 再承認後の親本文の進捗概要が未確認を表示し、旧完了通知の「残件なし」を現在の状態として出さないこと（既存テストは status だけを検査）。

## 同種の不具合の掃除（重要）
1と2は「機械記録の所在を親Issue番号と仮定する」同じ種類の不具合。3は「過去の記録を現在の状態として表示する」不具合。同じ前提が残る箇所を、リポジトリ全体（`.agents/ops/` 全体、skill・README中のURL例を含む）で検索して洗い出し、見つけたものは同じPRで修正してテストする。検索方法と、確認して問題なしとした箇所の一覧を報告する。もう一度の再レビューで同種が見つからないことが目標。

## 共通
- 指摘・同種の修正以外の変更はしない。関連テストと pre-commit を通し、push後の対象HEADのCI結果を報告する。
- 作業時間の上限は3時間。止めてよいのは、修正とテストを終えてCI結果が出たとき、または親の判断が必要な矛盾が見つかったとき（独立部分はコミット済みにして報告）。

## 報告
修正HEAD、各指摘の修正と回帰テストのnode ID、同種の掃除の検索方法・修正箇所・問題なしとした箇所、実行したテストと結果、変更した全パス、CI結果。

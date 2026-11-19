# #1964 期限切れclose（improvement-expire）が他の親の担当記録の更新で失敗する不具合の修正

担当Issue: https://github.com/Motoki0705/tennis-lab/issues/1964 （親自身を作業対象とする未分割Issue。子Issueは作らない）
worktree: `[local path]`（branch `codex/proposal-expiry-registry`、origin/main `5dcef6915` 起点）

#1964 の本体は PR #1971 で merge 済み（`bbe944308`）。merge後に親が日次timer `tennis-lab-improvement-expire` を導入した。
#1964 の要件は、本文の「全体方針・合意事項」の 2.（期限切れclose）と完了条件6。最初に本文と、再openコメント（2026-11-15）を読むこと。

## 観測（親が確認した事実）

- timer の初回実行（2026-11-15 00:00 JST）が非0終了した。closeしたIssueはない（#2007 は期限前）。出力:
  ```
  {"closed": [], "dry_run": false, "errors": ["parent registry changed during GitHub retrieval; retry the command to fetch current state"], "pending": []}
  ```
- merge直後の手動 `improvement-expire --dry-run`（checkout_job.py 経由）でも、3回中1回が同じエラーになった。
- 経路:
  - `parent/cli.py` の `improvement-expire` 分岐は、`store.locked()`（dry-runでは `store.snapshot()`）の中で `expire()` を呼ぶ。
  - `expire()` は focus なしの `ParentSession(gh, store, parent, None)` を元親ごとに作り、GitHub取得を繰り返す。
  - `parent/state.py` の取得中guard（240〜265行付近）は、focus が None だと registry 全体の差分をすべて conflict とみなす。
  - そのため、無関係な親の担当記録（heartbeat・委任・レビュー等）が取得中に1件でも更新されると失敗する。複数の親が常に動く運用では、期限切れcloseが繰り返し失敗し得る。

## 求める結果

- 他の親の registry 更新が並行しても、`improvement-expire` が期限切れの提案を close できる。dry-run も同様。
- 安全性は維持する:
  - 二重close・二重コメントをしない。
  - 取り出し送信済み・取り出し済みの提案を close しない。
  - GitHubの取得・書込み失敗では非0で終了する。
  - 再実行で同じ結果に収束する。
  - 期限判定はGitHubの作成時刻と出典目印だけで行う。
  - 同じrepoの propose・decide・extract・expire は、改善操作ロック（`store.operation(lock_key(repo))`）で直列化する。
- 修正方針は、既存契約を読んだうえで選ぶ。案:
  - (a) expire は registry の担当行を書かないので、registry 全体の transaction（`store.locked()`）に入れず、改善操作ロックだけで直列化する。
  - (b) focus のない読取でも、無関係な親の行の変化は許容し、expire が依存する状態の変化だけを conflict にする。
- どちらの案でも、expire が registry に書く必要があるかどうか（`append` や判断ログの記録経路を含む）を確認し、書くなら、その書込みが並行更新と安全に両立することを示す。
- 静かなフォールバック（conflict を無視して古い状態で書く、上限なしで再試行し続ける等）はしない。有限回の再試行で済ませる場合は、それが正しさに必要な不変条件を壊さない理由を示す。ただし、根本原因の解消を優先する。

## テスト

[tests README](tests/README.md#テスト作成方針) に従い、既存の expire テストと fake GitHub を再利用する。少なくとも次を検証する:
- 取得中に別の親の registry 行が更新されても、期限切れの提案を close する（修正を戻すと、今回のエラーで失敗するテスト）。dry-run も同様。
- 同時に同じ提案に対して extract／propose が動く場合に、改善操作ロックで直列化され、取り出された提案を close しない。
- 既存の expire テスト（期限前は何もしない、期限後にclose、再実行で変化なし、GitHub失敗で非0、取り出し済みは対象外）と、registry の conflict 検出を使う他コマンドのテストが退行しない。

関連テストを実行し、pushごとに対象HEADのCI `Python` を確認する。

## 成果

- コミットは pre-commit を通す。
- PRを作る。PR本文は日本語で、`Closes #1964` を付ける（親のレビューCLIが閉じる記述を要求するため）。
- 最終報告に含めるもの:
  - PR番号・HEAD
  - 原因の説明と、選んだ修正方針・理由
  - expire の registry 書込みの有無
  - 追加テストの名前と、修正を戻すと失敗することの確認
  - 関連テストの結果
  - CI結果

# PR #2423 正式レビュー（major）への対応

指摘2件を親が採用した。直したら、検証してcommit・pushし、PR番号と完全なHEADを宣言して終了する。CIの結果は後続のターンで受け取る。

## 1. 通知テンプレートの導入に失敗しても、timerを再び有効にできてしまう
`spin ops install storage-tier` が失敗しても、手順は本番cycleに進めてしまう。`spin ops on storage-tier` も通知unitを一度も確認せずに成功を返す。この状態で日次cycleが失敗すると、通知が出ない。
- コードで防ぐ。`spin ops on`（と `status`）は、対象のserviceが `OnFailure` で参照する通知テンプレートのインスタンスが `LoadState=loaded` であることを確かめる。確かめられなければ、有効化を拒否して理由を示す。静かに成功させない。同じ仕組みを使う他の対象でも効く形にする。
- PR本文のマージ後の手順も直す。install が失敗したら止める。有効化の前に、読み込まれたserviceの `OnFailure` と、通知インスタンスの `LoadState` を確認する。
- 拒否の経路を回帰テストで検査する。

## 2. 未送信の確定と送信結果不明を、journalで区別できない
`ToastNotSent`（確実に送れていない）と `ToastError`（送れたか不明）が、どちらも同じ `Windows toast failed` として記録される。
- 両方とも通知unitは非0で終わらせたまま、journalに「未送信（確定）」と「送信結果不明（重複の可能性あり）」を書き分ける。
- 既存の通知CLIテストで両方の例外を注入し、記録の違いを検証する。

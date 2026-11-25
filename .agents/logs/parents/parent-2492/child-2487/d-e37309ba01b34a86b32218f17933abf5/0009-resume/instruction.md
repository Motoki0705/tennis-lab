# #2487 rfdetr: queue のログを保持先へ移す

前回の報告（`f92c40d60`、RF-DETR-L の PT16・TRT16 × batch 1・2）を受け取った。記録を承認する。統合担当が #2499 に取り込む。

## queue のログを保持先へコピーする
`.training_queue/logs/` のログは、完了から7日で自動削除される（`repro/` は対象外）。PR #2500 の正式レビューで、ログが消えると再採点や可視化の読込が失敗することが指摘された。#2487 の記録にも `.training_queue/logs/` を直接参照しているものがある。
- あなたの branch の記録が `.training_queue/logs/` を参照していれば、そのログを自分の保持先（自分の出力ディレクトリ）に SHA-256 付きでコピーする。記録の参照と、それを読むコードを、コピー先に揃える。元の SHA-256 と一致することを確かめる。
- 親が #2492 関連のログ101件を `.worktree-outputs/parent-2492-queue-log-archive/logs/` に複製済み（`SHA256SUMS` 付き）。元ログがすでに消えていたら、そこから取ってよい。
- ログが消えた状態を模擬して、読込が保持先から動くことを確かめるテストを足す（読むコードがある場合）。
- これから入れる job のログも、完了後に同じようにコピーする。
- 対象になりうる記録: `run-meiji-player-model-comparison/rfdetr-speed-initial.json`、`run-meiji-rfdetr-large-accuracy/results.json`、`run-meiji-rfdetr-large-speed/results.json`。

## 終わり方
GPU は使わない。テストと pre-commit を通して push する。最終回答には、コピーしたログの件数と保持先、直した記録、push した HEAD を書き、`PARENT_RESULT {"schema_version":1,"state":"done","wait":null}` を返す。

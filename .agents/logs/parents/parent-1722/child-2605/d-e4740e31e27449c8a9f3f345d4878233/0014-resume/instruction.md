# 再開（自動再開の上限のため親が手動で再開）

宣言した local job `20261126T104356Z_5d9a31240786_scene-events-v3-training-compare` は終了済み。exit_code 0、所要 159.331 秒、終了 2026-11-26T10:46:36Z。log は `[local path]`（training_sha256 manifest e456f07c…、adopted_scenes 69、annotation_sha256_matches 69）。

宣言どおり照合結果を確認し、配布先の切替、PR #2610 の実行証拠、レビュー修正ごとの回帰テストの記載を更新する。コードを変えなければ `state: done` で終了する。新しい local job や CI 待ちを増やす場合は、親の手動再開が必要になるため、必要性を最小限にする。

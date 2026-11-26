# 再開（自動再開の上限に達したため、親が手動で再開する）

宣言した local job `20261126T095841Z_ca20e7872491_scene-events-v3-merged-head` は終了済み。exit_code 0、所要 2,463.93 秒、終了 2026-11-26T10:39:45Z。log は `[local path]`。

続きの作業:
- 宣言どおり、現配布版とのイベント配列・coverage のバイト照合とチェックサム検証を確認し、配布先と PR #2610 の実行証拠を HEAD `2999c0e743805fb2145ae6e03121f81b8c98df0c` で更新する。
- あわせて、後続の #2600 が学習に使った版 `data/tennis_multiview/processed/meiji_4cam/benchmarks/.multiview-v3-59a62db3c`（manifest SHA-256 `e456f07c…`）とも、69クリップの採用 scene.npz の全メンバー（イベント配列・coverage・v2 由来の配列）と分割をバイト照合し、結果を PR 本文に1行で書く。親は `e456f07c…` と `40efd634…` の間で、69件の scene.npz の SHA-256 と分割が一致することを確認済み。
- PR 本文の round 1 の major 3件それぞれに、修正 commit と回帰テストを対応づけて書く（未記載なら）。repo 内で完全な pipeline 設定を読む入口の分類結果も書く。
- コードを変えなければ CI 待ちは要らない。完了したら `state: done` で終了する。コードを変えた場合は push して CI を wait で宣言する。

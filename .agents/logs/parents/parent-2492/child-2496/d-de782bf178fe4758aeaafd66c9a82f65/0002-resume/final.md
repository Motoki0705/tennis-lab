CIで失敗した3件を修正し、同じブランチとPRへpushしました。FP32キャッシュ互換マニフェストが参照するNVMLコードを元の版に復元し、追加の計測をベンチマーク側へ移しました。計測用メソッドは `measure_forward()` に改名しました。 [PR #2497](https://github.com/Motoki0705/tennis-lab/pull/2497)

ローカルでは失敗した3件を含む58件が通過し、pre-commitのruffとmypyも通過しました。ブランチは `codex/parent-2492-issue-2496-primary`、新しい完全HEADは `d64962a28d81dd7646c7a19b33f88143b67a90f4` です。新HEADのCIと実GPU計測は確認待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2497,"head":"d64962a28d81dd7646c7a19b33f88143b67a90f4","resume_on_success":true,"next_step":"代表6 clipの固定frame・人物cropをIDとSHA-256付きで保存し、body_view_selectionと既存TensorRT資産を照合して、現行モデルのPT32/PT16/TRT16計測を共有training queueへ投入する。"}}
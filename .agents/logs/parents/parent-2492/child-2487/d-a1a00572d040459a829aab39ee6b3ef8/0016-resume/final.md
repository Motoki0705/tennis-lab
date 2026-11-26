EdgeCrafter の queue ログ26件を `.worktree-outputs/parent-2492-issue-2487-edgecrafter/player-model-comparison/queue-logs/` に保存し、元ログと SHA-256 が全件一致しました。JSON 8件の参照を更新し、queue 履歴がない一時環境で26件のログと記録間の6参照を読み直しました。pytest 93件、pre-commit、knowledge 検証が通過しました。[検証記録](https://github.com/Motoki0705/tennis-lab/issues/2487#issuecomment-7166644840)

TensorRT の調査を終了しました。元の `trt16` と `trt16-fp32attn` の実測値を「変換の不具合、原因未特定」として保存し、EdgeCrafter の Pareto 判定に使う条件は PT16 と記録しました。最初に出力がずれる演算とバッチ内の画像順への依存は、採用候補になった場合の残課題です。[EDGECRAFTER.md](https://github.com/Motoki0705/tennis-lab/blob/2d05a6de50103f4903e073a58fa5c0a71ae61645/knowledge/runs/run-meiji-player-model-comparison/EDGECRAFTER.md)

`codex/parent-2492-issue-2487-edgecrafter` に `2d05a6de50103f4903e073a58fa5c0a71ae61645` を push し、CI を起動しました。統合時は `comparison-evidence.json` の EdgeCrafter 参照 SHA-256 3件の更新が必要です。[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45643545436) · [統合 PR #2499](https://github.com/Motoki0705/tennis-lab/pull/2499)

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}
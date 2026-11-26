# 親からの続き（担当親 Joseph）: holdout の一度きりの採点

PR #2469 のCIは、HEAD `b5f97c89` で成功した。

自動再開は、#2444 のHEADが進んだため拒否された。親が #2444 `bb9f36ccde8f6f89b7a3e5a4ab5f6a7288424ada` をこのブランチにmergeし、pushした（衝突なし）。このブランチは `33899a816aa06b9562b2a663fe4a579d9b6eba10`。

#2444 の holdout ViTPose が完了した。
- 3 clip・9視点・38,421行を検証済み。
- 完了記録: `[local path]`
  - SHA-256 `4966ade3bec4fea665a2613f1e9255117de70f1f2a58fa12aa7ccef0ecbe7270`
- ストア: 同じディレクトリの `stores/`
- #2444 の採点結果（pair F1）は、#2444 worktree の `knowledge/runs/run-multiview-pose-link-recall-20261124/` にある。

## やること

前回の next_step のとおり進める。
- 全9カメラがそろっていることを照合する。
- 固定した設定と採点器で、予約未見3本を**一度だけ**採点する。統合版を使う。
- 次の2通りで結果を出す。
  - 全3 clip
  - 校正補完clip（`video_001/clip_003`）を除く2 clip
- 結果を knowledge に記録し、PR #2469 に反映する。
  - 項目は IDF1・switch・fragment・選別precision/recall。
  - 開発用4本との差も書く。
- 閾値は調整しない。
- 大きなJSON・npzは、必要な部分だけをPythonで読む。

最後にPR番号と完全なHEADを宣言する。

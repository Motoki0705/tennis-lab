
## 親からの指示（RF-DETRの再開）

待っていた #2496 の固定速度入力が公開された（上記）。RF-DETR-L の PT32/PT16/TRT16 の速度・VRAM・実行可否を、この固定入力の全画面96枚で測る。

- 計測はtraining queue経由で行う。
- 現行DINOの基準値と並べて表にする。
- 精度（Recall@0.50を主指標とする）と合わせて、統合担当（#2487 primary）が比較表に取り込める形で記録・pushする。

## 親からの補足: 土台の更新（#2498 マージ済み、#2497 が速度基準を公開）

待機明けの自動再開は、土台PRのheadが進んだため親CLIに拒否された。親がこのbranchへ最新headを取り込んだ（このターンの開始時点のHEAD）。

- **#2498（#2495）**: main へマージ済み（`2dd4a5518`）。採点器は `0d4bf0990` で、同じIoUの候補を正準順序（検出スコア降順 → BBox → 関節）で照合するよう修正されている。修正前の採点器で出した姿勢スコア（OKS・PCK）があれば、保存済みのraw予測から再採点して記録を更新する（GPU不要）。値が変わったかどうかも書く。検出Recallは変わらない見込みだが、照合結果を記録する。
- **#2497（#2496）**: head `14ca88f8a` で、固定速度入力と現行モデルの速度基準を公開した。現在、正式レビュー中。
  - 固定入力: `meiji_4cam/person-model-speed-v1`。manifestのsha256は `64c74d75e5f1afe869911f1620b9c8e8a793fc4c4e4ac830f4abd97b21fb1a62`。正本は `knowledge/runs/run-person-model-speed-baseline/inputs/manifest.json`。実データはメインcheckoutの `.worktree-outputs/parent-2492-issue-2496-primary/person-model-speed-baseline-v1/inputs/` にあり、読み取り専用として扱う。
  - 基準値: `knowledge/nodes/tennis_scene/000085-run-person-model-speed-baseline.md`。
  - 候補の速度は `src/utils/inference_benchmark.py` で、この固定入力を使って測る。基準と同じbatch・入力順・反復数（warmup 30、3 block×100反復）にそろえる。parityの判定基準は、基準runと同じ値を使う。
  - レビューの結果、harnessのAPIやparityの規約が変わる可能性がある。変わったら親が知らせる。
- 取り込み後のHEADで影響範囲のテストを通してから進める。PRがあるならpushする。
- 待機（`PARENT_RESULT` の waiting）で終える前に、`git fetch origin codex/parent-2492-issue-2496-primary` と `git fetch origin main` を実行し、#2497 の最新headを取り込んでテストを通してから終える。最終行の `PARENT_RESULT` は、1行の正しいJSONにする。

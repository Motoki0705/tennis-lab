---
id: run-blcs-inference-checkpoint-migration
type: run
task: blcs
sequence: 69
recorded_at: '2026-10-14'
date: '2026-10-14'
title: 既存BLCS重みの明示的metadata移行と厳密な推論復元
provider: codex
status: done
issue: 1044
config:
  migration:
    model.physics_head: false
  predict_velocity: false
  training_loss_and_metrics_preserved: true
  device: cpu
  dtype: float32
metrics:
  maximum_position_difference_m: 0.0
  identical_state_dict_tensors: 131
artifacts:
  run_dir: knowledge/runs/run-blcs-inference-checkpoint-migration
  checkpoint: /home/kamimura/projects/tennis-lab/ckpt/blcs/axial-base-physics-v3-kp14-t128-v3-6-e200-s42-epoch189-inference-v1.ckpt
  checkpoint_sha256: 1ca51e312d881ade6d635263b271ba2e253c2dbb3ffac0cdde490e4697be5bf4
  original_checkpoint: /home/kamimura/projects/tennis-lab/ckpt/blcs/axial-base-physics-v3-kp14-t128-v3-6-e200-s42-epoch189.ckpt
  original_checkpoint_sha256: 67e01836b1016ca1d0fb2acf088bfe44aa49a804c91aa8fb9cc132ec115dd197
  verification: knowledge/runs/run-blcs-inference-checkpoint-migration/migration.json
  log: knowledge/runs/run-blcs-inference-checkpoint-migration/migration.log
  migration_script: knowledge/runs/run-blcs-inference-checkpoint-migration/migrate.py
  original_prediction: {status: deleted, record: knowledge/runs/run-blcs-inference-checkpoint-migration/artifact-manifest.json#/files/6}
  migrated_prediction: {status: deleted, record: knowledge/runs/run-blcs-inference-checkpoint-migration/artifact-manifest.json#/files/5}
  original_review: knowledge/runs/run-blcs-inference-checkpoint-migration/claude-review.json
  review_response: knowledge/runs/run-blcs-inference-checkpoint-migration/review-response.md
  verification_checks: knowledge/runs/run-blcs-inference-checkpoint-migration/checks.json
  reproduction: knowledge/runs/run-blcs-inference-checkpoint-migration/repro.sh
parents:
- group-i1044-progressive-losses-u4000
relations: []
papers: []
tags:
- inference
- checkpoint-migration
- no-retraining
repro:
  commit: 08ea733650aeb374f10871043313bc031a3b36a2
  branch: codex/issue-1044-blcs-physics
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: >-
    CUDA_VISIBLE_DEVICES='' .venv/bin/python
    knowledge/runs/run-blcs-inference-checkpoint-migration/migrate.py
    --source /home/kamimura/projects/tennis-lab/ckpt/blcs/axial-base-physics-v3-kp14-t128-v3-6-e200-s42-epoch189.ckpt
    --output /home/kamimura/projects/tennis-lab/ckpt/blcs/axial-base-physics-v3-kp14-t128-v3-6-e200-s42-epoch189-inference-v1.ckpt
    --original-checkout /home/kamimura/projects/tennis-lab
    --evidence-dir /home/kamimura/projects/tennis-lab/.claude/worktrees/issue-1044-blcs-physics/knowledge/runs/run-blcs-inference-checkpoint-migration
---

既存のepoch189重みを現在の通常推論で実行できるよう、`model.physics_head=false`を明記した別checkpointを作成した。元checkpointはそのまま保持し、学習済みtensor、既存の`predict_velocity=false`、旧loss／metrics metadataは変更していない。新しい実験モデルの採用・本番モデルの再学習ではない。

通常推論は厳密なモデル設定とstate dictからモデルだけを復元し、学習用Lightning module・loss・metricsの初期化に依存しない。未知のstate prefix、欠損・余分なkey、shape不一致は拒否する。UIも同じモデルmetadata検証を行い、未移行checkpointには`physics_head`欠落の理由を表示する。`single_object-...`という名前付きsnapshotもsingle_objectとして認識する。移行版のファイル名は末尾が`-inference-v1.ckpt`で、元の参考重みと区別する。

元artifactのSHA-256を固定値と照合し、保存し直した131個のstate tensorがすべて完全一致することを確認した。元checkout `9f5fed173b9dd56e4a1bb72622ff2c771b8d1f66` の公開`BLCSPredictor.load_from_checkpoint`と、新版の同じ公開APIをそれぞれCPUで実行した。固定seed1044・2view・8frameの同一入力について、物理mの位置出力は配列単位で完全一致し、最大差は0 mだった。これはschema移行の同値性確認であり、test性能の再評価ではない。元の10run比較値・選定・採用保留は変わらない。

初回は保存済みOmegaConfのstruct制約により新key追加前に失敗した（`attempt1.log`）。追加を明示的な`open_dict`内で行うよう修正し、再実行した。移行版はtensor一致と両公開APIの出力一致を確認した後に公開し、元fileは上書きしていない。再現時の`--original-checkout`は上記の元revisionを指す必要がある。`--output`には未作成の別pathを指定する。

この検証は学習runではないためTensorBoard曲線はない。保存した予測・SHA台帳と実データの厳密復元テストを根拠とする。

現行コードの再現revisionは上記commitで固定し、一度だけ使う移行scriptはこのbundleに保持する。再現用`repro.sh`は元・新版それぞれのcheckoutを明示指定し、revision不一致や既存の出力先を拒否する。既存の作業checkoutを切り替えたり、保存済み結果を上書きしたりしない。レビュー6指摘への判断・修正は[対応記録](../../runs/run-blcs-inference-checkpoint-migration/review-response.md)にまとめた。ユーザー指示に従い、この変更後のClaude再レビューは起動していない。

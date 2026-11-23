# Person feature split reference

`person-features-before-split.npz` was generated with the pre-split
`FeatureExtractor` and `PersonTrackingModule` at commit `f7b016700`.
It contains 64 frames, 122 original detection rows and deterministic CPU model
outputs. Rows are sparse, poses reverse the ordering of two geometrically
identical people after frame 10, frames 15/16 and the trailing frame are empty.
The fake pose preserves raw confidence 1.4; the unprompted CLIP fake returns a
unit vector in the adopted 1280-dimensional space. AFLink returns identity
links. Real StrongSORT, GSI and court selection run without substitution.

`person-tracking-before-split.npz` and `person-selection-before-split.json`
freeze every tracking evidence array, GSI coordinates/masks, selection/group
axes and selection diagnostics produced by those original components.
The reference has IDs 1/2, four interpolated positions and 118 selected real
observations. Tests compare exact values after reloading the new feature
artifacts. Real model/GPU acceptance is separate from this CPU contract.


# FP32キャッシュ互換テスト

`main_fp32_cache.tar.gz` は、共有court sideを取り込んだmain `dcd6534594f73cca12a45c2d9afaa8ac09d48bc7` の
`TennisSceneOrchestrator` と `ClipStore` が保存した3camera・24frameのfixture。
既存の `test_auto_pipeline.py` の数値と仮の重みを使い、モデル推論を固定の出力に置き換えている。
他のディレクトリで開けるよう、入力動画・重みのidentityに相対パスを使う。
`baseline.json` は同じ実行の公開identityとソースファイルのSHA-256を保持する。

このfixtureはC3以前の追跡v5・選別v2を保持する。C3ではencoderの完全identityを追跡証拠へ保存するため、
追跡v6・選別v3への再生成を要求する。過去fixtureは変更せず、この拒否条件を検証する。
既定FP32と明示FP32の再利用には、固定main `603823383cd941941556cafdcbc98ff67a19f51f` の
コミット済みソースから計算したcode digestを使う。現行schemaの小型storeとdatasetの公開マーカーをそのdigestで保存し、
現行のorchestratorで既存componentをexecute/loadから再利用し、公開済みannotationを `overwrite=false` で受理できることを調べる。
`phase2_main_source_digests.json.gz` の `sources` は `git archive` で取り出した固定mainの全 `src/**/*.py` のSHA-256。
`revised_sources` は同じ固定mainから `revised_commit`（`a42c9f6e75ea982af391010280161ef353cd9dd5`）までに変わった
PythonファイルのSHA-256。両コミット間の `git diff --name-only` と導入時の `git show` から生成する。
テストはこの2時点の指紋で互換パッチを再現し、台帳などの対象ファイルに後続の編集が入った場合の非互換も検査する。

過去fixtureを再生成する場合は、このテストを含むworktreeから実行する。

```bash
worktree=$(pwd)
baseline=$(mktemp -d)
scratch=$(mktemp -d)
git archive dcd6534594f73cca12a45c2d9afaa8ac09d48bc7 src tests/unit/tennis_scene/pipeline/test_auto_pipeline.py |
    tar -x -C "$baseline"
cd "$baseline"
PYTHONPATH="$baseline" "$worktree/.venv/bin/python" \
    "$worktree/tests/integration/tennis_scene/test_fp32_cache_compatibility.py" \
    "$scratch" "$worktree/tests/integration/tennis_scene/fixtures/main_fp32_cache.tar.gz"
```

# 実RGB観測・再構成の診断

CLIは `src.tennis_scene.scripts`、処理と型付き設定はこのディレクトリに置く。
出力は `paths.output_root` 配下の `tennis_scene/evaluate/<experiment>/<run-id>`。
既存runを再利用せず、入力の教師・動画・checkpointは変更しない。
rootは各設定の `paths`、その配下はrole相対fragmentで指定する。
GPUを使う診断は [training queue](../../../../.agents/skills/training-queue/SKILL.md) 経由で実行する。

| CLI (`python -m` の後に指定) | 設定・目的 |
|---|---|
| `src.tennis_scene.scripts.benchmark_vitpose_precision` | [`benchmark_vitpose_precision.yaml`](../../configs/benchmark_vitpose_precision.yaml)。保存boxを固定してFP32/BF16の座標・confidence差と時間を比較。`clip`はDATA、`observations`はOUTPUT、`checkpoint`はEXTERNAL_ASSET相対。 |
| `src.tennis_scene.scripts.probe_meiji_court` | [`probe_meiji_court.yaml`](../../configs/probe_meiji_court.yaml)。OUTPUT配下の指定Court checkpointとfull/ball cropを比較。手動Courtは評価targetにだけ使用し、cropは外注ボール観測から作る。fit拒否は理由とraw値を保存。 |
| `src.tennis_scene.scripts.evaluate_refinement` | [`evaluate_refinement.yaml`](../../configs/evaluate_refinement.yaml)。OUTPUT相対の`scene`/`court`を読み、指定build recipeのrefinement設定で補正前後を評価。GPU再推論なし。coverage不合格は結果保存後もエラーを返す。 |

共通設定・crop・精度の値はリンク先YAMLを正本とする。
Court/refinementの `recipe` と `overrides` は同じscene configディレクトリの設定をcomposeする。
`paths` は診断側を唯一のroot指定元とするため、recipeの `overrides` へpath rootを含めない。
Court/ViTPoseは動画のFPS・サイズ・frame数をmanifestと照合し、不整合で停止する。

CPUでrefinementを監査する例:

```bash
.venv/bin/python -m src.tennis_scene.scripts.evaluate_refinement \
  scene=tennis_scene/generate/experiment/run/scene.npz \
  court=tennis_scene/precompute/experiment/run/court.npz
```

CPU画像レビューは `src.tennis_scene.scripts.render_reconstruction_review`。
`--mode teachers` は `--dataset-root` / `--run-root` と繰り返し `--clip` を受け、
raw/refined教師を比較する。`--mode frames` は `--data-root` / `--dataset` /
`--video` / 単一 `--clip` / `--frames` と任意 `--cameras` で指定frameを描画する。
両modeとも絶対 `--output-root` と相対
`--output tennis_scene/visualize/<experiment>/<run-id>` が必要。
既存のmode省略呼び出しは、解析済みroot引数から一意に選択する。
異なるmodeの引数を混ぜるとエラーになる。完全な引数は `--help` で確認できる。

BLCSの固定test評価は [BLCS README](../../../tasks/blcs/README.md) を参照。

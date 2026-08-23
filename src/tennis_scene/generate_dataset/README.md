# `src/tennis_scene/generate_dataset`

`clip_studio` が追記した構造化データセットを読み、`TennisSceneOrchestrator` の出力をクリップ単位の疑似アノテーションとして保存します。データセット全体を一度に作り直さず、`annotation.json` がないクリップだけを処理します。

## 生成物

```text
<dataset_root>/
├── dataset.json
└── clips/<recording_id>/<clip_name>/
    ├── clip.json
    ├── media/<camera_id>.mp4
    └── annotations/tennis_scene/
        ├── scene.npz
        ├── scene.metadata.json
        ├── annotation.json
        └── pipeline_config.yaml
```

`scene.npz` は `SceneResult` のcanonical schemaです。`player_position` / `ball_3d`は常にcourt/worldの`[m]`で、sidecarの`court_coordinate_normalization`は生成元PLCS/BLCSのversion/scale provenanceです。単一正本は `src/utils/schema/court_normalization.py` と `src/tasks/base/data/court_coordinate_contract.py` であり、provenanceによって公開arrayを再scaleしません。`annotation.json` は必須配列のshape/dtypeと入力 `clip.json` のSHA-256を持つ完成マーカーで、これがないディレクトリは完成済みとして扱いません。既存の完成結果は既定でskipし、再生成は `overwrite=true` でのみ行います。失敗は `annotations/tennis_scene.failure.json` に記録し、CLIは非0で終了します。

## 実行

```bash
# まだ生成されていない全クリップ
.venv/bin/python -m src.tennis_scene.scripts.generate_dataset \
  dataset_directory=tennis_scene/dataset

# 一部だけ明示選択
.venv/bin/python -m src.tennis_scene.scripts.generate_dataset \
  dataset_directory=tennis_scene/dataset \
  clip_ids='[match1/clip_000]'

# v2はv1 datasetを上書きしない別rootへ生成する
.venv/bin/python -m src.tennis_scene.scripts.generate_dataset \
  court_coordinate_normalization=v2 \
  dataset_directory=tennis_scene/dataset_norm_v2
```

モデル・checkpoint設定は既存の `configs/pipeline.yaml` を直接読み、`configs/generate_dataset.yaml` はdataset生成時の差分だけを `pipeline_overrides` として保持します。共有normalization groupはgenerate boundaryからnested pipelineへ渡され、PLCS/BLCS checkpoint・保存済みstage result・最終SceneResultが同じversion/scaleであることを検証します。metadataを持たない旧checkpoint/stage resultは明示的な`v1` runtimeでのみ使用でき、`v2`では生成前にエラーになります。これによりパイプライン設定を二重管理しません。

BLCSのreprojection lossに必要な実カメラparameterはこのschemaでは捏造しません。実データを既存のsimulation dataset loaderへ混ぜる処理は、キャリブレーション契約とsplit方針を決めた後の別スコープです。

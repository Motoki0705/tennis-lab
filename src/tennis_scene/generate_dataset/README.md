# 構造化clipへの疑似アノテーション

clip_studioの同期clipを標準TennisSceneOrchestratorへ渡し、SceneResult v2と品質maskを
保存します。モデル・座標・欠測契約は[tennis_scene README](../README.md)が正本です。
生成設定は[generate_dataset.yaml](../configs/generate_dataset.yaml)、stage設定は
[pipeline.yaml](../configs/pipeline.yaml)を使います。

```bash
# GPU実行は共有training queueへ登録する。
.venv/bin/python -m src.tennis_scene.scripts.generate_dataset \
  dataset_directory=tennis_multivew/processed/meiji_3cam/dataset \
  'clip_ids=[video_000/clip_000]'
```

checkpoint変更はpipeline_overridesで指定します。手動side・Court・人物対応artifactは
不要です。元clip全体を保存するため、この入口でmax_framesによる切詰めはできません。

```text
<dataset>/videos/<video>/clips/<clip>/annotations/
├── tennis_scene/
│   ├── scene.npz
│   ├── scene.metadata.json
│   ├── annotation.json
│   └── pipeline_config.yaml
└── tennis_scene.failure.json       # 失敗時
```

annotation.jsonは最後に公開する完成markerです。scene schema、shape/dtype、処理status、
有効frame数、clip.json・動画・設定/重みの識別情報を持ちます。
同一入力・同一設定の完成結果だけをskipし、変更時はoverwrite=trueが必要です。
不完全なtransactionは完成済みとして扱いません。

partialやemptyもmaskとともに保存します。教師に使える範囲は下流が3D validityで判定します。
失敗はfailure markerへ記録し、continue_on_errorに従って次へ進みます。失敗があればCLIは非0終了です。

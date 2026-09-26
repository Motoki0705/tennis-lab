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

成果物はclip内の`annotations/tennis_scene/`に保存します。
component成果物と統合exportの配置は[pipeline仕様](../pipeline/README.md#成果物)を参照してください。
`annotation.json`はdataset向けの完成markerで、`scene_result`に採用したimmutableなNPZ exportを記録します。
scene schema、shape/dtype、status、有効frame数、clip.json・動画・設定/重みの識別情報を検証します。
同一入力・同一設定の完成結果だけをskipし、変更時はoverwrite=trueが必要です。
公開時にcomponent storeを削除・置換しません。

partialやemptyもmaskとともに保存します。教師に使える範囲は下流が3D validityで判定します。
失敗はfailure markerへ記録し、continue_on_errorに従って次へ進みます。失敗があればCLIは非0終了です。

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

checkpoint変更はpipeline_overridesで指定します。Courtの手動入力は不要ですが、
人物対応とsideはモデル実装が無いため（#933 / #932）、確認済みartifactがclip storeに必要です
（[pipeline仕様](../pipeline/README.md#処理単位)）。元clip全体を保存するため、この入口でmax_framesによる切詰めはできません。

成果物はclip内の`annotations/tennis_scene/`に保存します。
component成果物と統合exportの配置は[pipeline仕様](../pipeline/README.md#成果物)を参照してください。
`annotation.json`はdataset向けの完成markerで、`scene_result`に採用したimmutableなNPZ exportを記録します。
公開するのはdeclared pipelineのscene v2だけです。scene schema、shape/dtype、status、有効frame数、
clip.json・動画・設定/重みの識別情報（publication identity）を検証します。
展開済みpipeline設定は`configs/<SHA-256>.yaml`に内容アドレスで保存し、exportには書き込みません。
同一入力・同一設定の完成結果だけをskipし、変更時や識別情報の無い旧markerではoverwrite=trueが必要です。
公開時にcomponent storeを削除・置換しません。旧layout（markerが`scene.npz`を直接指すv1）の既存データは
reader側が引き続き読みます。

partialやemptyもmaskとともに保存します。教師に使える範囲は下流が3D validityで判定します。
失敗はfailure markerへ記録し、continue_on_errorに従って次へ進みます。失敗があればCLIは非0終了です。

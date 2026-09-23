# Ball Detection

出力先と実験ごとの設定方針は [タスク出力規約](../OUTPUTS.md) を参照。

`src/tasks/ball_detection` は、RGBフレーム列から各フレーム内のテニスボール位置を推定するタスク実装です。モデル定義・データ取り込み(TrackNet/YouTube/Web統合ストア)・学習・推論・評価・可視化・データセット生成までを一貫して提供します。

## Modules

### (ルート)
- **`__init__.py`**: 検証済みmodel+adapterを返す `build_ball_detection_pair` のパッケージ入口。

### models/
- **`__init__.py`**: model実装とdiscriminator factoryの公開面。
- **`spatiotemporal_unet.py`**: `SpatioTemporalUNet`。`(B,C,T,H,W)→(B,1,T,H/2,W/2)`、`T>=8` 必須の時空間 U-Net。
- **`conv_next_unet.py`**: `ConvNeXtUNet`。ConvNeXt ブロックベースの spatio-temporal U-Net(`T>=1` で動作)。
- **`dinov3_rope.py`**: `DINOv3RoPEBallDetector`。model I/O境界で準備済みのDINOv3 patch token・RoPE周波数・attention maskを3軸RoPE decoderで処理するRGB専用ヒートマップ検出器。
- **`discriminators/__init__.py`**: `build_ball_detection_discriminator(config)` の工場関数。

### model_io/
- **`contracts.py`**: RGB入力、model call、学習batch、typed predictionの契約。
- **`adapters.py`**: 推論の生RGBはfloat32・有限値・`[0,1]`を検証し、checkpointの正規化後にRGB/MDD・layout変換を行う。学習・評価のdataset側で正規化済みの入力は宣言されたchannel範囲を検証してそのまま使い、二重に正規化しない。loss/output decodeも担当。DINOv3ではraw backbone応答の検証、patch token decode、RoPE周波数とattention maskの生成もこの境界で完了する。
- **`factory.py`**: `model.name` (`stunet`/`conv_next_unet`/`dinov3_rope`) からmodel+adapterを一度だけ選択し、DINOv3 backboneのfrozen/trainable実行経路も構築時にbind。
- **`evaluation.py`**: checkpointから検証済みpairを読み、評価loopへprobability heatmapを提供。

### data/
- **`__init__.py`**: `build_ball_detection_datamodule(config)`。`data.source` からDataModuleを選択。
- **`types.py`**: `FrameLabel`/`ClipWindow`/`BallDetectionSample`/`BallDetectionBatch` のデータ契約。
- **`dataset.py`**: `BallDetectionDataset`。`ClipWindow` をモデル入力サンプル(画像・heatmap・座標)へ変換する共通実装。
- **`tracknet_datamodule.py`**: `TrackNetDataModule`。TrackNet形式(`Label.csv`+連番jpg)を読む。
- **`youtube_datamodule.py`**: `YouTubeDataModule`。`TrackNetDataModule` を継承しYouTube split解決だけ変更。
- **`web_datamodule.py`**: `WebBallDataModule`。`data/tennis/web/unified` 統一ストアを読む(`static`/`temporal`モード)。
- **`staged_datamodule.py`**: `StagedBallDataModule`(issue #579)。TrackNet+Webを混合し可変長 `T` で学習。
- **`components/augmentation.py`**: `BallDetectionAugmentation`。回転/flip/affine/crop/色/ノイズ/ゼロマスク等の augmentation 合成。
- **`components/staged_sampler.py`**: 可変 `T` 用バッチサンプラー群(`VariableTBatchSampler` 等)。
- **`components/web/data_access_layer/web_store.py`**: `WebFrameStore`。統一ストアの read-only アクセサ。
- **`components/web/data_access_layer/writer.py`**: shard書き込み・index構築・アトミック publish を行う writer 群。
- **`components/web/parser/*.py`**: Roboflow/RacketVision/Kaggle/Ball-YOLO 各データセットの parser。

### training/
- **`lightning_module.py`**: `BallDetectionLightningModule`。Focal損失によるヒートマップ学習、GAN併用可。
- **`metrics.py`**: `BallDetectionMetrics`。ハンガリアン対応付けによる `precision`/`recall`/`f1`/`mean_distance_px`。
- **`runner.py`**: `BallDetectionTrainingRunner`。datamodule/lightning_module構築の薄いアダプタ。
- **`staged_calibration.py`**: `probe_batch_size_by_t()`。`T` ごとのOOM較正でバッチサイズを決定。
- **`staged_lightning_module.py`**: `StagedBallDetectionLightningModule`。手動最適化による可変T勾配蓄積学習。
- **`staged_runner.py`**: `StagedBallDetectionTrainingRunner`。フェーズ間のOOM較正とweightのみ引き継ぎを制御。

### inference/
- **`checkpoint.py`**: predictor・レビューUI共通の推論専用loader。保存されたmodel設定・`model.`重みと入力正規化をstrict復元する。`data.augmentation.normalize_imagenet.enabled`は必須で、有効なら保存されたmean/stdも使う。学習専用オプションは要求・補完しない。
- **`predictor.py`**: `BallDetectionPredictor`。checkpointのadapterを維持し、CPU上の `BallPrediction(coords, confidence, heatmaps)` を返す。

### evaluation/
- **`contracts.py`**: 評価マニフェスト(`ball_detection_evaluation_manifest_v1`)の型付き契約。
- **`configuration.py`**: checkpoint設定読み出しとモデル名整合性検証。
- **`dataset_provenance.py`**: データセットの provenance(ハッシュ・ソース)記録。
- **`metrics.py`**: `StratifiedBallMetrics`。全体/データソース別のメトリクス追跡。
- **`evaluator.py`**: 1 job(checkpoint×dataset×split) を評価する `DefaultJobEvaluator`。
- **`reporting.py`**: `summary.json`/`comparison.csv`/`comparison.md` を生成。
- **`runner.py`**: `EvaluationPipeline`。fingerprintベースの再利用付き複数job評価。

### visualization/
- **`orchestrator.py`**: checkpointからのスライディングウィンドウ推論→GIF保存を統括。
- **`adapters/predict_inputs.py`**: スライディングウィンドウ開始位置とバッチ構築。
- **`adapters/render_inputs.py`**: MDDフレーム/学習バッチの描画用変換。
- **`api/predict.py`**: `predict_clip()`。重複ウィンドウ推論の集約と `PredictionSequence` 構築。
- **`io/clip.py`**: クリップディレクトリから推論/描画用テンソルを構築。
- **`rendering/clip_renderer.py`**: RGB/MDD/予測/heatmapの2x2グリッド描画。
- **`review/datasets.py`**: `BallDatasetCatalog`。TrackNet/YouTube/unified webを走査し、シーン(opaque ID)・dense frame位置・multi-instance `FrameLabel` を提供する。
- **`review/checkpoints.py`**: `scan_checkpoints()`。checkpoint本体の保存configから `model.name`・`num_frames`・窓下限・metrics既定を読む。
- **`inference/loader.py`**: `load_ball_model()`。共通checkpoint loaderを使い、レビュー用の入力サイズ・窓長を検証する。
- **`inference/peaks.py`**: `decode_frame_peaks()`。canonicalなthreshold/NMS/top-k + subpixel refineで複数peakをoriginal image pixelへ写す。
- **`inference/rasters.py`**: 予測probability heatmapのRGBA overlay。
- **`inference/service.py`**: `DetectionService`。catalog/scenes/preview/image/validate/inferを提供する共有Webバックエンド。

### generate_dataset/
- **`candidate_workflow.py`**: 候補区間の手動選択(`run_candidate_selection`)と疑似ラベル推論(`predict_candidates`)。
- **`annotation_session.py`**: 疑似ラベルレビューOpenCV UIと確定処理(`finalize_candidate`)。

### scripts/
- **`train.py` / `train_staged.py`**: 通常 / staged 学習エントリポイント。
- **`eval.py`**: 単一checkpointの詳細診断評価。
- **`evaluate_manifest.py`**: manifestベースの複数checkpoint比較評価。
- **`visualize.py`**: クリップ単位のGIF可視化生成。
- **`convert_web_dataset.py`**: web生データセット群を統一ストアへアトミック変換。
- **`analyze_web_bbox_ratio.py`**: bbox最大辺比率の分布解析。
- **`preview_augmentation.py` / `preview_heatmaps.py`**: augmentation / heatmap生成の確認用プレビュー。
- **`youtube/*.py`**: YouTube動画取得・候補選択・疑似ラベル推論・アノテーション確定・DINOv3 SSL画像収集の各スクリプト。

### configs/
- モデル/データ/損失・メトリクス/学習/staged学習フェーズ/評価マニフェスト/可視化ごとにHydra設定を分割。

## データセットレビュー / 推論UI

起動コマンドと操作手順は[Web UI利用ガイド](visualization/README.md)、HTTP APIは
[共有Detection UI](../base/visualization/detection/README.md)を参照してください。ここには
ball_detection固有のsourceと互換契約だけを記す。

### source

| id | 実体 | mode | 備考 |
|---|---|---|---|
| `tracknet` | `data/tennis/tracknet/<game>/<Clip*>/Label.csv` + 連番jpg | temporal | 既定source |
| `youtube` | `data/tennis/youtube/frames/<video>/<clip_*>/Label.csv` + 連番jpg | temporal | YouTubeアノテーション |
| `web_static` | `data/tennis/web/unified` の `temporal=0` sample | static | 1 frame = 1 scene |
| `web_temporal` | 同ストアの `temporal=1` sample(sequence単位) | temporal | frame順は保存 `frame_index` |

`web_static`/`web_temporal` は unified store(`index.npz`)が未生成なら
catalogに `available=false` と理由を出すだけで、空の一覧を捏造しない。
scene IDは `"<dataset>::<scene>"` で、HTTP層はこれをcatalogの列挙結果として
解決する。任意pathを受け取るAPIは提供しない。

### 表示・推論の契約

- GTは保存済み `FrameLabel`(multi-instance、`visibility`付き)を
  original image pixelの `x,y` で返す。補間・3D化・推定ラベルは行わない。
  rastersは予測probability heatmapのみで、GT ball Gaussianは捏造しない。
- アノテーション行が無いframeは「missing」であり、明示的なnegativeとは区別する。
  previewは `annotated=false` と warningを返し、推論metricsはそのframeを除外する
  (`metrics.scored_frames` / `excluded_frames`)。窓全体が未アノテーションなら
  `metrics.available=false` と理由を返し、0埋めのmetricを捏造しない。
  非finiteな座標・visibilityは読み込み時に拒否し、JSONへNaNを出さない。
- 推論窓の長さは checkpointの `model.num_frames` を上限とし、下限は
  アーキテクチャ最小(`stunet`=8、その他=1)にMDD multi-frame時の2 frame要件を
  加えた値。範囲外はpadせず422で拒否する。
- static sourceは unified storeの正規static sampling(1 frameを
  `model.num_frames` 回反復)だけを使い、`metrics.window.mode="static_repeat"`
  とwarningで明示する。反復した窓の予測は平均heatmapへ明示的に集約して
  **単一の元frame**としてdecode・採点し、itemsのindexを一意にする
  (同一GTを反復回数だけ数えない)。集約前の反復回数は `metrics.window.repeat`
  に残す。temporal sourceは選択frame以降の連続窓のみで、シーン長を超える要求は
  拒否する。
- モデル入力は original frameを checkpointの `data.image_size` へ
  `INTER_LINEAR` でresizeした float32 `[0,1]` RGBを渡す。保存された入力正規化は
  model I/O境界で一度だけ適用する。MDD変換は `model_io/adapters.py` の境界で行い、UI側では再実装しない。
- metricsは `BallDetectionMetrics` をそのまま使い、Hungarian matchingと
  checkpoint保存の `ball_distance_threshold`(original pixel)で採点する。
  一致検出が0件の平均距離は `null` (UIではN/A) とし、誤差0とは表示しない。

### 更新と境界

- `catalog()` は呼び出しごとにデータ・checkpoint rootを再走査する。追加された
  clipや `*.ckpt` は再起動なしで一覧に出て、利用できないsourceの理由は
  2回目以降の呼び出しでも失われない。checkpointの窓・しきい値などの契約も
  常に同じ応答内の情報から決まる。
- `validate` / `infer` は実行直前にcheckpointの `(size, mtime_ns)` を確認し、
  本体が差し替わっていれば保存configを読み直してから使う。旧metadataで別の
  checkpointを推論しない。catalogが一度拒否したcheckpointは再読込で復活させず、
  復旧は `catalog()` の再走査で行う。
- 読み取りはconfigured root内に限定する。root外へ解決される `*.ckpt`
  symlinkは `error` 付きで拒否し、root外を指すclipディレクトリやframe
  symlinkは対象sceneを除外してwarningに残す。unified storeの `paths` が
  configured data root外を指す場合もstoreを unavailable として理由を返す。
  正規writerの `../source/image.jpg` のような原画像参照はdata root内なら許可する。

### checkpoint互換

checkpoint本体の保存configだけを根拠にする(ファイル名から推論しない)。

- `model.name` が `stunet`/`conv_next_unet`/`dinov3_rope` 以外、または
  `num_frames < アーキテクチャ最小` のcheckpointは `error` 付きで一覧に出し、
  実行時に明示的に失敗させる。
- `model.input_mode` か入力 `image_size` が欠落したcheckpointも同じく
  `error` 付きで早期に unusable とする。
- metricsは**キー欠落**なら `configs/metrics/default.yaml` の既定値を使って
  warningに残し、**値が不正**(範囲外・`nan`/`inf`・型違い)なら
  checkpointを unusable にする(黙って別のしきい値へ置き換えない)。
- dataset互換性は学習時の `data.source` ではなくアーキテクチャ制約で決める。
  static sourceは正規反復モードがあるため常に実行可能、temporal sourceは
  最小窓以上の場合だけ互換とする。`scenes(checkpoint=)` はdataset単位の互換性に
  加えて**sceneごとのframe数**で絞り、短すぎるclipを実行候補に出さない。

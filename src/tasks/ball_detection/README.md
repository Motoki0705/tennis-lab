# Ball Detection

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
- **`adapters.py`**: forward前にfloat32・有限値・`[0, 1]`を含む入力契約を検証し、RGB/MDD・layout変換とloss/output decodeを担当。DINOv3ではraw backbone応答の検証、patch token decode、RoPE周波数とattention maskの生成もこの境界で完了する。
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

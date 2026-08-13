# Court Detection

テニス映像からコート構造 (keypoint / segmentation / line) を推定するタスク群です。`data.task` (`kp`/`seg`/`line`) によってデータセット・モデル出力チャネル数・損失・メトリクスが切り替わる統一パイプラインになっています。

## Modules

### (root)
- **`__init__.py`**: 検証済みmodel+task adapterを返す `build_court_detection_pair` のパッケージ入口。

### data/
- **`datamodule.py`**: `CourtDetectionDataModule`。`task` に応じてdataset/collateを切替え8の倍数へパディング。
- **`court_kp_dataset.py`**: `CourtKPDataset`。14点keypointをGaussianヒートマップに変換。
- **`court_seg_dataset.py`**: `CourtSegDataset`。`masks/{id}.png` を持つサンプルのみ使用。
- **`court_line_dataset.py`**: `CourtLineDataset`。白線二値maskを読む。
- **`augmentation.py`**: `build_seg_transforms()`/`build_kp_transforms()`。task別augmentationパイプライン。

### models/
- **`__init__.py`**: `CourtHierarchicalModel` とencoder/decoder実装の公開面。
- **`hierarchical_model.py`**: `CourtHierarchicalModel`。encoder→decoder→1x1convの既定アーキテクチャ。
- **`encoders.py`**: `CourtDefaultEncoder`/`CourtDINOv3Encoder`。既定CNN encoderと、model I/O境界から呼び出すDINOv3 backboneの構築情報を提供。
- **`decoder.py`**: `CourtFPNDecoder`/`CourtUNetDecoder`/`CourtDPTDecoder`。

### model_io/
- **`contracts.py`**: kp/seg/lineの入力・学習batch・typed prediction契約。
- **`adapters.py`**: forward前にImageNet正規化済みfloat32 RGBの有限値・値域を検証し、task固有loss・prediction decodeを担当。DINOv3ではraw中間token応答の検証と4段階feature mapへの変換もこの境界で完了する。
- **`factory.py`**: task-model channel整合性を検証し、model+adapterとDINOv3 backboneのfrozen/trainable実行経路を構築時にbind。
- **`images.py`**: 3predictor共通のuint8 RGB検証・resize・ImageNet正規化境界。

### training/
- **`lightning_module.py`**: `CourtDetectionLightningModule`。選択済みadapterへ入力/loss/output decodeを委譲。
- **`losses.py`**: task-localな `DiceLoss`/`BinaryDiceLoss`。
- **`metrics.py`**: `CourtDetectionMetrics`。task別に mIoU/mean_dist/Dice を算出。
- **`runner.py`**: `CourtDetectionTrainingRunner`。薄いアダプタ。

### inference/
- **`predictor.py`**: `CourtKeypointPredictor`。semantic classごとのmulti-peakを `CourtKeypointPrediction(keypoints, scores, valid, covariance, heatmaps)` として返す。covarianceはoriginal-image pixel座標のlocal heatmap momentである。
- **`mask_predictor.py`**: task別の `CourtSegPredictor`/`CourtLinePredictor`。typed dense predictionを返す。

### evaluation/
- **`homography_quality.py`**: 14点アノテーションへRANSACホモグラフィーを当て、inlier被覆・再投影誤差・可視率・占有率・地上視点の射影歪みを評価。
- **`image_evidence.py`**: 投影した9本のコートラインに対する画像edge支持率と、色・明度・重複確認用descriptorを算出。
- **`pipeline.py`**: `data_train.json`互換JSONを厳密に読み、複数データセットの採否manifestと補正済み14点JSONを出力。

### visualization/
- **`orchestrator.py`**: 選択済みtask pipelineの共通render APIからGIF保存を統括。
- **`adapters/`**: predictor入力変換と学習時qualitative描画用変換。
- **`api/predict.py`**: factoryでkp/seg/line pipelineを一度だけ選択し、分岐なしのframe loopを提供。
- **`io/frames.py`**: `CourtFrame`/`load_court_frames()`。
- **`rendering/`**: task別2パネル描画(`kp_renderer`/`seg_renderer`/`line_renderer`)と共通style(`common.py`)。

### scripts/
- **`train.py`**: 学習エントリポイント。
- **`visualize.py`**: 2-panel GIF可視化。
- **`generate_masks.py`**: 14点keypointから6 court cellのsegmentation maskを生成。
- **`generate_line_masks.py`**: 14点keypointから白線maskを生成。
- **`preview_heatmaps.py`**: `sigma_ratio` 比較プレビュー。
- **`prepare_youtube_dataset.py`**: YouTube動画取得〜20点アノテーション雛形生成。
- **`annotate_youtube_keypoints.py`**: 20点アノテーションUIエントリポイント。
- **`evaluate_homography_annotations.py`**: Hydra設定の複数JSONを一括してホモグラフィー品質評価。

### generate_dataset/
- **`annotation_session.py`**: `run_annotation_session()`。CourtKP20の手動/pseudo-labelアノテーションUI本体。

### configs/
- data(task別+augmentation)・model(hierarchical + encoder/decoder)・loss・training(default/lora)・run・visualization の各Hydra設定。DINOv3+DPTは `model/encoder=dinov3 model/decoder=dpt` で選択。

**注意**: 学習は14点keypointのみ使用する一方YouTubeアノテーションは20点収集しており、変換経路は本ディレクトリ内に見当たらない。`configs/visualization/*.yaml` に `versino_0` というtypoあり。

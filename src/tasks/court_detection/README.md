# Court Detection

テニス映像からコート構造 (keypoint / segmentation / line) を推定するタスク群です。`data.task` (`kp`/`seg`/`line`) によってデータセット・モデル出力チャネル数・損失・メトリクスが切り替わる統一パイプラインになっています。

## Modules

### (root)
- **`__init__.py`**: 3タスク(`kp`/`seg`/`line`)の定義と `CourtHierarchicalModel` の re-export。

### data/
- **`datamodule.py`**: `CourtDetectionDataModule`。`task` に応じてdataset/collateを切替え8の倍数へパディング。
- **`court_kp_dataset.py`**: `CourtKPDataset`。14点keypointをGaussianヒートマップに変換。
- **`court_seg_dataset.py`**: `CourtSegDataset`。`masks/{id}.png` を持つサンプルのみ使用。
- **`court_line_dataset.py`**: `CourtLineDataset`。白線二値maskを読む。
- **`augmentation.py`**: `build_seg_transforms()`/`build_kp_transforms()`。task別augmentationパイプライン。

### models/
- **`__init__.py`**: `build_court_detection_model(config)`。task-model num_classes整合性を検証する工場関数。
- **`hierarchical_model.py`**: `CourtHierarchicalModel`。encoder→decoder→1x1convの既定アーキテクチャ。
- **`encoders.py`**: `CourtDefaultEncoder`/`CourtDINOv3Encoder`。CNN特徴またはDINOv3中間トークンを4段階特徴として返す。
- **`decoder.py`**: `CourtFPNDecoder`/`CourtUNetDecoder`/`CourtDPTDecoder`。

### training/
- **`lightning_module.py`**: `CourtDetectionLightningModule`。task別に損失・メトリクス・予測保存形式を切替。
- **`losses.py`**: `DiceLoss`/`BinaryDiceLoss`(`FocalBCEWithLogitsLoss` は base から re-export)。
- **`metrics.py`**: `CourtDetectionMetrics`。task別に mIoU/mean_dist/Dice を算出。
- **`runner.py`**: `CourtDetectionTrainingRunner`。薄いアダプタ。

### inference/
- **`predictor.py`**: `CourtKeypointPredictor`。heatmap argmaxを元画像座標のkeypointへ復元。
- **`mask_predictor.py`**: `CourtSegPredictor`/`CourtLinePredictor`。dense prediction共通テンプレート。
- **`preprocess.py`**: `preprocess_court_image()`。3predictor共通の前処理。

### visualization/
- **`orchestrator.py`**: task別にpredict→render→GIF保存を統括。
- **`adapters/`**: predictor入力変換と学習時qualitative描画用変換。
- **`api/predict.py`**: `predict_kp/seg/line()`。
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

### generate_dataset/
- **`annotation_session.py`**: `run_annotation_session()`。CourtKP20の手動/pseudo-labelアノテーションUI本体。

### configs/
- data(task別+augmentation)・model(hierarchical + encoder/decoder)・loss・training(default/lora)・run・visualization の各Hydra設定。DINOv3+DPTは `model/encoder=dinov3 model/decoder=dpt` で選択。

**注意**: 学習は14点keypointのみ使用する一方YouTubeアノテーションは20点収集しており、変換経路は本ディレクトリ内に見当たらない。`configs/visualization/*.yaml` に `versino_0` というtypoあり。

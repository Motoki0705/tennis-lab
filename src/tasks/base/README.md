# `src/tasks/base`

各 task パッケージ (`plcs`, `blcs` 等) が再利用する学習・データ入出力・推論・可視化の共通基盤です。抽象クラスは共通のライフサイクル/検証ロジックを持ち、task 固有の部分だけをサブクラスの override ポイントへ委譲する設計です。

## Modules

### Top-level
- **`__init__.py`**: data/training/inferenceの共有APIを再export。
- **`preview.py`**: dataset previewスクリプト共通helper(`resolve_split_file`/`resolve_sample_indices`)。

### data/
- **`scene_dataset.py`**: `SceneDatasetBase`。シーンディレクトリ読込・window/camera選択・`build_sample()`拡張点を持つDataset基底。
- **`datamodule.py`**: `SceneDirectoryDataModule`。固定 `scene_dir`+split txt を扱うDataModule基底。
- **`chunked_datamodule.py`**: `BaseChunkedDataModule`。trainのみバックグラウンド生成chunkに切替え。
- **`chunk_manager.py`**: `ChunkManager`。chunk生成スレッドのライフサイクル管理。
- **`dataset_writer.py`**: `BaseDatasetWriter`。npy+jsonシーン書き出しの共通実装。
- **`augmentation.py`**: `BaseObservationAugmentation`。augmentation config解析・dispatchガードの共通部分。
- **`court_lines.py`**: 投影済み CourtKP20 から共有`court_line_map`を描画するbuilderと、preview/診断用のRANSAC有限線分抽出を分離して提供する。map-space augmentationはconfigで明示的に切り替える。

### training/
- **`lightning_module.py`**: `BaseLightningModule`。optimizer/scheduler構築とqualitative/test予測保存の拡張点。
- **`runner.py`**: `BaseTrainingRunner`。configをsingle source of truthとする学習実行の共通フロー。
- **`chunk_rotation_callback.py`**: `ChunkRotationCallback`。epoch終端でchunked datamoduleを回転。
- **`gan_training.py` / `gan_loss.py` / `gan_transition_callback.py`**: 手動最適化ベースのGAN学習共通実装(`LSGANLoss`含む)。
- **`qualitative_callback.py` / `qualitative_saving.py`**: validationサンプルの可視化描画・GIF/画像保存。
- **`losses.py`**: `FocalBCEWithLogitsLoss`。複数taskで重複していた実装を統合。

### inference/
- **`predictor.py`**: `BasePredictor`。checkpoint解決・device解決・共通predict契約(CPU tensor/snake_case)を持つ基底。

### generate_dataset/
- **`parallel_runner.py`**: `run_parallel_scene_generation()`。`ProcessPoolExecutor`(spawn context)による並列シーン生成fan-out。

### visualization/
- **`frames.py`**: 画像ソース読込(`load_rgb_frames`等)。
- **`gif.py`**: `save_gif()`。共通GIF writer。
- **`io.py`**: `BaseSceneBundle`/`resolve_cameras()`。カメラ選択解決の共通ロジック。
- **`layout.py`**: マルチパネル合成ジオメトリ(`compose_row`/`compose_grid`等)。
- **`orchestrator.py`**: `BaseVisualizationRuntimeConfig`/`build_scene_runtime_config()`。可視化オーケストレータ共通スキャフォールド。
- **`style.py`**: `SceneStyleConfig`/`parse_scene_style()`/`parse_view_3d()`。BLCS/PLCS共通の `visualization.style`(theme/影/トレイル/HUD/ミニマップ)と `visualization.view_3d`(共有3D視点)のtyped parse。未知キー・未知テーマはエラー。

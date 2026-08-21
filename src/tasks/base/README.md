# `src/tasks/base`

各 task パッケージ (`plcs`, `blcs` 等) が再利用する学習・データ入出力・推論・可視化の共通基盤です。抽象クラスは共通のライフサイクル/検証ロジックを持ち、task 固有の部分だけをサブクラスの override ポイントへ委譲する設計です。

## Modules

### Top-level
- **`__init__.py`**: data/training/inferenceの共有APIを再export。

### model_io/
- **`contracts.py`**: task-local adapterが実装するtyped `ModelIOAdapter`、immutableな`ModelCall`、model/adapterをcomposition時に一度だけ検証する`bind_model_io()`/`BoundModelIO`。tensor keyやshapeの意味は各taskが所有する。
- **`tensors.py`**: `TensorSpec`/`require_tensor()`。dtype/rank/fixed dimensionを`forward`前のadapter boundaryで検証する。

### data/
- **`scene_dataset.py`**: `SceneDatasetBase`。シーンディレクトリ読込・window/camera選択・`build_sample()`拡張点を持つDataset基底。`meta.num_frames` は正の整数として必須で、payload長との矛盾は `SceneDataContractError` によりindex作成前に失敗する。
- **`datamodule.py`**: `SceneDirectoryDataModule`。固定 `scene_dir`+split txt を扱うDataModule基底。
- **`chunked_datamodule.py`**: `BaseChunkedDataModule`。trainのみバックグラウンド生成chunkに切替え。
- **`chunk_manager.py`**: `ChunkManager`。chunk生成スレッドのライフサイクル管理。
- **`dataset_writer.py`**: `BaseDatasetWriter`。npy+jsonシーン書き出しの共通実装。
- **`augmentation.py`**: `BaseObservationAugmentation`。augmentation config解析・dispatchガードの共通部分。
- **`canonical_tracking.py`**: tracking sceneのclip/view選択と可変 `(V,T,D)` paddingを担う共通Dataset基盤。
- **`lifecycle_slots.py`**: birth/death区間をinterval coloringで固定query数へ詰め、death後のslot再利用教師を生成。

### training/
- **`lightning_module.py`**: `BaseLightningModule`。optimizer/scheduler構築、明示的なcompile target契約、qualitative/test予測保存の拡張点。
- **`compilation.py`**: `compile_modules()`。primary modelやGAN discriminatorなど、名前付きtargetへidentity/state_dictを保つ`nn.Module.compile()`を適用する。重複参照はobject identityで一度だけ処理し、失敗時にeagerへfallbackしない。
- **`runner.py`**: `BaseTrainingRunner`。configをsingle source of truthとする学習実行の共通フロー。`init_weights`読込後・`Trainer`構築前に全compile targetを処理する。
- **`chunk_rotation_callback.py`**: `ChunkRotationCallback`。epoch終端でchunked datamoduleを回転。
- **`gan_training.py` / `gan_loss.py` / `gan_transition_callback.py`**: 手動最適化ベースのGAN学習共通実装(`LSGANLoss`含む)。
- **`qualitative_callback.py` / `qualitative_saving.py`**: validationサンプルの可視化描画・GIF/画像保存。
- **`losses.py`**: `FocalBCEWithLogitsLoss`。複数taskで重複していた実装を統合。
- **`tracking_lifecycle.py`**: active/inactive/birth/death近傍を重み付けするpresence BCE。
- **`tracking_lightning_module.py`**: BLCS/PLCSに共通するtracking stage dispatch、loss/metric logging、test prediction収集・保存を所有し、task固有adapter/loss/metrics/payloadはhookへ委譲する。
- **`tracking_metrics.py`**: lifecycle segment単位のbirth/death誤差・presence F1・query再利用・ID switch診断。

## Training model compilation

全taskのtraining configは次の共有契約を明示し、標準ではcompileを有効にする。

```yaml
compile:
  enabled: true
  backend: inductor
  mode: default
  fullgraph: false
  dynamic: false
```

`BaseLightningModule.compilation_targets()`は`self.model`をprimary targetとして返す。primary model内で呼ばれる子moduleは個別登録しない。GAN有効時は`ManualGANSupportMixin`が独立して呼ばれる`discriminator`を追加する。将来teacher/studentなどを追加するtaskは`additional_compilation_targets()`で名前付き`nn.Module`を明示する。loss/metricを含む全childの自動探索は禁止する。

標準modeはCUDA Graphsを暗黙に有効化しない`default`とする。`reduce-overhead`と`max-autotune`は明示選択でき、共有Lightning lifecycleがouter batch境界をCUDA Graphsへ通知する。ただしmodel forward内部のgraph breakをまたぐtensor lifetimeはmodel依存であるため、CUDA Graphs modeの互換性は選択したtask/modelで検証する。

`run.dry_run=true`はmodelを構築しないためcompileしない。`run.fast_dev_run=true`は通常どおりcompileする。`run.resume`はin-place compile後も元のstate-dict keyで復元し、`run.init_weights`はweight読込後にcompileする。staged ball detectionのOOM calibrationも同じ設定でprobe modelをcompileする。設定不正、target契約不正、compile失敗は例外として伝播し、暗黙のeager fallbackは行わない。

### inference/
- **`predictor.py`**: genericな`BasePredictor[PredictionT]`。checkpoint/device解決を共有し、predictのdecoded型は選択済みtask adapterが所有する（decoded result内のtensorはCPU）。明示CUDA指定はfallbackせず、availability選択は`auto`だけが行う。

### generate_dataset/
- **`parallel_runner.py`**: `run_parallel_scene_generation()`。`ProcessPoolExecutor`(spawn context)による並列シーン生成fan-out。
- **`timeline_composer.py`**: BLCS/PLCS共通の固定global timelineへsource subclipを配置し、同時存在数とlifecycle metadataを保証。

### visualization/
- **`preview.py`**: dataset previewスクリプト共通helper(`resolve_split_file`/`resolve_sample_indices`等)のcanonical owner。
- **`frames.py`**: 画像ソース読込(`load_rgb_frames`等)。
- **`gif.py`**: `save_gif()`。共通GIF writer。
- **`io.py`**: `BaseSceneBundle`/`resolve_cameras()`。カメラ選択解決の共通ロジック。
- **`layout.py`**: マルチパネル合成ジオメトリ(`compose_row`/`compose_grid`等)。
- **`orchestrator.py`**: `BaseVisualizationRuntimeConfig`/`build_scene_runtime_config()`。可視化オーケストレータ共通スキャフォールド。
- **`style.py`**: `SceneStyleConfig`/`parse_scene_style()`/`parse_view_3d()`。BLCS/PLCS共通の `visualization.style`(theme/影/トレイル/HUD/ミニマップ)と `visualization.view_3d`(共有3D視点)のtyped parse。未知キー・未知テーマはエラー。

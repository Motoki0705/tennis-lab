# Ball Detection

出力先と実験ごとの設定方針は [タスク出力規約](../OUTPUTS.md) を参照。

`src/tasks/ball_detection` は、RGBフレーム列から各フレーム内のテニスボール位置を推定するタスク実装です。モデル定義・データ取り込み(ball frame store/Web統合ストア)・学習・推論・評価・可視化・データセット生成までを一貫して提供します。

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
- **`candidates.py`**: 閾値0で局所peakを抽出し、native格子のpatchと境界maskを返す。採点用の閾値やtrajectory gateは適用しない。
- **`adapters.py`**: 推論の生RGBはfloat32・有限値・`[0,1]`を検証し、checkpointの正規化後にRGB/MDD・layout変換を行う。学習・評価のdataset側で正規化済みの入力は宣言されたchannel範囲を検証してそのまま使い、二重に正規化しない。loss/output decodeも担当。DINOv3ではraw backbone応答の検証、patch token decode、RoPE周波数とattention maskの生成もこの境界で完了する。
- **`factory.py`**: `model.name` (`stunet`/`conv_next_unet`/`dinov3_rope`) からmodel+adapterを一度だけ選択し、DINOv3 backboneのfrozen/trainable実行経路も構築時にbind。
- **`evaluation.py`**: checkpointから検証済みpairを読み、評価loopへprobability heatmapを提供。

### data/
- **`__init__.py`**: `build_ball_detection_datamodule(config)`。`data.source` からDataModuleを選択。
- **`store.py`**: `BallFrameStore`。TrackNet・Meiji・chat_annotation を統一した frame store(`data/ball_detection/<version>`、`ball_detection_frames.v1`)の読み出しと検証。clip の全frameを JPEG shard + 列指向 `index.npz` で保存し、`point_kind`・`segment_break`・`event` などのラベル意味論の正本。
- **`types.py`**: `FrameLabel`/`BallDetectionSample`/`BallDetectionBatch` のデータ契約。
- **`dataset.py`**: `BallDetectionDataset`。store/Web の窓を画像・heatmap・座標・教師マスクへ変換する共通実装。
- **`store_datamodule.py`**: `BallStoreDataModule`。storeのsplitを読み、`train_sampling.source_weights` の比率で窓を混合する。`null` は全窓を1回ずつ読む。
- **`store_dataset.py`**: clip内の連続窓を読む。短いclip・教師がない窓の除外数をsource別に記録する。
- **`supervision.py`**: point_kindから教師マスクを決める。既定の正例はobserved、負例はレビュー済みでinstanceなし／out_of_frameのみ。unresolved・interpolated・occlusion_estimated・未レビューはframe全体をloss/metricsから除外する。
- **`web_datamodule.py`**: `WebBallDataModule`。`data/tennis/web/unified` 統一ストアを読む(`static`/`temporal`モード)。
- **`staged_datamodule.py`**: `StagedBallDataModule`(issue #579)。ball store+Webを混合し可変長 `T` で学習。
- **`components/augmentation.py`**: `BallDetectionAugmentation`。回転/flip/affine/crop/色/ノイズ/ゼロマスク等の augmentation 合成。
- **`components/staged_sampler.py`**: 可変 `T` 用バッチサンプラー群(`VariableTBatchSampler` 等)。
- **`components/web/data_access_layer/web_store.py`**: `WebFrameStore`。統一ストアの read-only アクセサ。
- **`components/web/data_access_layer/writer.py`**: shard書き込み・index構築・アトミック publish を行う writer 群。
- **`components/web/parser/*.py`**: Roboflow/RacketVision/Kaggle/Ball-YOLO 各データセットの parser。

### training/
- **`lightning_module.py`**: `BallDetectionLightningModule`。Focal損失によるヒートマップ学習、GAN併用可。
- **`metrics.py`**: `BallDetectionMetrics`。ハンガリアン対応付けによる `precision`/`recall`/`f1`/`mean_distance_px`。
- **`runner.py`**: `BallDetectionTrainingRunner`。datamodule/lightning_module構築と、2D detectorのweights-only初期化。3D court metadataを要求せず、module全体の重みをstrictに復元する（GAN有効時のdiscriminatorも含む）。不完全な重み転送は拒否する。
- **`staged_calibration.py`**: `probe_batch_size_by_t()`。`T` ごとのOOM較正でバッチサイズを決定。
- **`staged_lightning_module.py`**: `StagedBallDetectionLightningModule`。手動最適化による可変T勾配蓄積学習。
- **`staged_runner.py`**: `StagedBallDetectionTrainingRunner`。フェーズ間のOOM較正とweightのみ引き継ぎを制御。

### inference/
- **`checkpoint.py`**: predictor・レビューUI共通の推論専用loader。保存されたmodel設定・`model.`重みと入力正規化をstrict復元する。`data.augmentation.normalize_imagenet.enabled`は必須で、有効なら保存されたmean/stdも使う。学習専用オプションは要求・補完しない。
- **`predictor.py`**: `BallDetectionPredictor`。checkpointのadapterを維持し、CPU上の `BallPrediction`（点・score・native heatmap・候補の局所特徴）を返す。

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
- **`adapters/render_inputs.py`**: 学習バッチの正規化契約を受け、RGB表示は逆正規化し、MDD表示は学習と同じ正規化済み画像から生成する。
- **`api/predict.py`**: `predict_clip()`。重複ウィンドウ推論の集約と `PredictionSequence` 構築。
- **`io/clip.py`**: storeのclipから推論/描画用テンソルを構築（`visualization.store_dir` と `clip_id` を指定）。
- **`rendering/clip_renderer.py`**: RGB/MDD/予測/heatmapの2x2グリッド描画。
- **`review/datasets.py`**: `BallDatasetCatalog`。ball storeの全versionとunified webを走査し、シーン(opaque ID)・dense frame位置・multi-instance `FrameLabel` を提供する。
- **`review/checkpoints.py`**: `scan_checkpoints()`。checkpoint本体の保存configから `model.name`・`num_frames`・窓下限・metrics既定を読む。
- **`inference/loader.py`**: `load_ball_model()`。共通checkpoint loaderを使い、レビュー用の入力サイズ・窓長を検証する。
- **`inference/peaks.py`**: `decode_frame_peaks()`。canonicalなthreshold/NMS/top-k + subpixel refineで複数peakをstored image pixelへ写す。
- **`inference/rasters.py`**: 予測probability heatmapのRGBA overlay。
- **`inference/service.py`**: `DetectionService`。catalog/scenes/preview/image/validate/inferを提供する共有Webバックエンド。

### generate_dataset/
- **`frame_store/`**: 統一 frame store の生成。`sources/{tracknet,meiji,chat_annotation}.py` が各注釈形式を検証して `ClipSpec`(`clip.py`)へ写し、`builder.py` が split 割当・JPEG shard 化・アトミック publish を行う。設定は `configs/generate_dataset.yaml`(`config.py` で厳密検証)、入口は `scripts/generate_dataset.py`。
- **`candidate_workflow.py`**: 候補区間の手動選択(`run_candidate_selection`)と疑似ラベル推論(`predict_candidates`)。
- **`annotation_session.py`**: 疑似ラベルレビューOpenCV UIと確定処理(`finalize_candidate`)。

### scripts/
- **`generate_dataset.py`**: 統一 frame store の生成エントリポイント。
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

## 検出証拠の出力契約

`BallPrediction` はCPU tensorで返す。従来の `coords` / `confidence` はnative heatmapの
argmax（任意のsubpixel補正付き）、`heatmaps` はモデルの出力格子のsigmoid値
`(B,T,H,W)` で、全格子をfloat32のまま保持する。scoreは未較正であり、球の存在確率ではない。

`candidates: BallCandidates` は次を追加する。KとPは呼び出し側が渡す
`BallCandidateConfig`、pipelineの既定値は
[`ball_detection.candidates`](../../tennis_scene/configs/pipeline.yaml)を参照。

| field | shape | 意味 |
|---|---|---|
| `coords` | B,T,K,2 | x/(W−1), y/(H−1)。argmaxと同じsubpixel補正 |
| `scores` / `valid` | B,T,K | 格子のsigmoid値 / 局所peakが存在するmask |
| `cells` | B,T,K,2 | 補正前の整数格子位置(x,y) |
| `patches` / `patch_valid` | B,T,K,P,P | cells中心のnative heatmap値 / 実格子内mask |

局所特徴は学習済みbackbone embeddingではなく、候補周辺のprobability patchである。
共有 `heatmaps_to_peaks` のcontrastive NMSを閾値0で使い、score順で最大K候補を残す。
弱いpeakも残り、平坦な複数画素のmapは `nms_kernel>1` なら候補0件になる。
同score候補間の順位は意味を持たない。候補不足は0埋め＋`valid=false`、
patchの画像外部分は0埋め＋`patch_valid=false` で区別する。
patchをsubpixel座標へ再sampleしないため、中心値は常にscoreと一致する。
元のdense heatmapも残るので、候補外の情報を使う処理や再抽出が可能。
pipelineでの座標変換・保存・単一点の受理は[pipelineの契約](../../tennis_scene/pipeline/README.md#ball検出証拠)を参照。

## データセットレビュー / 推論UI

起動コマンドと操作手順は[Web UI利用ガイド](visualization/README.md)、HTTP APIは
[共有Detection UI](../base/visualization/detection/README.md)を参照してください。ここには
ball_detection固有のsourceと互換契約だけを記す。

### source

| id | 実体 | mode | 備考 |
|---|---|---|---|
| `store/<version>` | `data/ball_detection/<version>` | temporal | TrackNet・Meiji・chat_annotation。1 camera-clip = 1 scene |
| `web_static` | `data/tennis/web/unified` の `temporal=0` sample | static | 1 frame = 1 scene |
| `web_temporal` | 同ストアの `temporal=1` sample(sequence単位) | temporal | frame順は保存 `frame_index` |

`web_static`/`web_temporal` は unified store(`index.npz`)が未生成なら
catalogに `available=false` と理由を出すだけで、空の一覧を捏造しない。
scene IDは `"<dataset>::<scene>"` で、HTTP層はこれをcatalogの列挙結果として
解決する。任意pathを受け取るAPIは提供しない。

### 表示・推論の契約

- GTは保存済み `FrameLabel`(multi-instance、`visibility`付き)を
  stored image pixelの `x,y` で返す。補間・3D化・推定ラベルは行わない。
  rastersは予測probability heatmapのみで、GT ball Gaussianは捏造しない。
- previewの `annotated` はレビューの有無、`supervised` はobserved-only方針で採点できるかを表す。
  未レビューおよび未確定・推定ラベルのframeはwarningを返し、推論metricsから除外する
  (`metrics.scored_frames` / `excluded_frames`)。窓全体が教師対象外なら
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
  symlinkは `error` 付きで拒否し、root外を指すstoreやshard
  symlinkはstoreを unavailable にして理由に残す。unified storeの `paths` が
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

## Storeへの移行

通常の学習は `data.source=store`、既定versionは `ball-mix-v1`。3 sourceの混合比と
教師方針は `configs/data/rgb_sequence.yaml` で明示する。`eval_stride: null` は
`model.num_frames` ごとの窓を意味する。旧source値 `tracknet` / `youtube` /
`mixed_tracknet` と旧splitテキストは受け付けない。stagedの設定名は `sources.store`。
Web統合ストアは従来の形式で読み、全frameを教師ありとして共通batch契約に合わせる。

TrackNetの配布形式を読むコードはstore生成時のsource adapterだけに限定する。
YouTube手動注釈ツールは `clip.json` を保存するが、旧学習用CSVは出力しない。
YouTubeの既存データとWebのstore変換はこの移行の対象外。
旧checkpointの推論は保存済みmodel/正規化契約のまま利用できる。学習再開には新しい
store設定を使い、旧data設定への自動フォールバックは行わない。

### Meiji混合FT

`configs/train_meiji_mixed.yaml` はft-e13からのweights-only FTを定義する。
モデル・入力前処理は既存設定を引き継ぎ、storeの3 sourceを等比率で混ぜる。
epoch budget・seed・学習率・checkpoint選択条件は同configを正本とする。
Meijiのsplitはstoreのmetadataに従い、教師方針は `data.supervision` に従う。

元repoのdata/checkpoint/output rootを明示し、共有training queueから実行する:

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.train --config-name train_meiji_mixed \
    paths.project_root=<worktree> paths.data_root=<元repo>/data \
    paths.checkpoint_root=<元repo>/ckpt paths.output_root=<元repo>/outputs \
    paths.artifact_root=<元repo>/outputs paths.cache_root=<元repo>/.cache \
    paths.external_asset_root=<元repo>/third_party
```

validationの窓はcheckpoint選択用であり、全frameのholdout評価とは区別する。
`test_after_fit=false` により最終epochの自動testを止め、選択済みcheckpointとft-e13を
同じ全frame・同じ復号条件で別途比較する。比較完了まではdeployを更新しない。

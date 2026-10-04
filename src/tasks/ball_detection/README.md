# Ball Detection

出力先と実験ごとの設定方針は [タスク出力規約](../OUTPUTS.md) を参照。

`src/tasks/ball_detection` は、RGBフレーム列から各フレーム内のテニスボール位置を推定するタスク実装です。モデル定義・データ取り込み(ball frame store)・学習・推論・評価・可視化・データセット生成までを一貫して提供します。

## Modules

### (ルート)
- **`__init__.py`**: 検証済みmodel+adapterを返す `build_ball_detection_pair` のパッケージ入口。

### models/ と model_io/
- **ConvNeXtUNet**: 2ch MDDからnative probability heatmap・候補・局所patchを出す。`model_io/factory.py`の検証済みmodel/adapter経由で使う。
- **[MDD＋pose coordinate detector](models/mdd_pose/README.md)**: 高解像度MDDの局所3D encoder、pose集約、時間RoPE/cross-attention、座標直接回帰。32frame・32条件の学習前レビュー実装。
- **`model_io/mdd.py`**: 共通の正負輝度差＋sigmoid変換。ConvNeXtの保存済みMDD係数・正規化契約は維持する。
- **`model_io/adapters.py`**: 元RGBの検証/宣言済み正規化→MDDへの単一経路、heatmap学習・復号。RGBを直接受け取るモデル分岐はない。
- **`model_io/contracts.py` / `candidates.py`**: heatmap・候補のtyped契約と閾値前局所peak抽出。
- **`models/discriminators/`**: ConvNeXtの任意GAN学習用discriminator。

STUNetとball用DINOv3 RoPE、専用設定・LoRA学習経路は削除済み。
対応しない旧checkpointは明示エラーにする。court等の共有DINOv3部品は対象外。
新モデルの座標をheatmapと偽るadapterは設けず、refiner接続は#986で別途決める。

### data/
- **[プレイ区間・学習窓の固定](data/PLAY_INTERVALS.md)**: `play_intervals.py`、`play_manifest.py`、`pose_windows.py`。approved pose subsetからプレイ/非プレイ候補と32frame窓を作り、位置教師不足を分離して可視化する。
- **`__init__.py`**: `build_ball_detection_datamodule(config)`。`data.source=store` を検証し、`BallStoreDataModule` を構築。
- **`store.py`**: `BallFrameStore`。TrackNet・Meiji・chat_annotation を統一した frame store(`data/ball_detection/<version>`、`ball_detection_frames.v1`)の読み出しと検証。clip の全frameを JPEG shard + 列指向 `index.npz` で保存し、`point_kind`・`segment_break`・`event` などのラベル意味論の正本。
- **`types.py`**: `FrameLabel`/`BallDetectionSample`/`BallDetectionBatch` のデータ契約。
- **`dataset.py`**: `BallDetectionDataset`。storeの固定長窓を画像・heatmap・座標・教師マスクへ変換する共通実装。
- **`store_datamodule.py`**: `BallStoreDataModule`。storeのsplitを読み、`train_sampling.source_weights` の比率で窓を混合する。`null` は全窓を1回ずつ読む。
- **`store_dataset.py`**: clip内の連続窓を読む。短いclip・教師がない窓の除外数をsource別に記録する。
- **`supervision.py`**: point_kindから教師マスクを決める。既定の正例はobserved、負例はレビュー済みでinstanceなし／out_of_frameのみ。unresolved・interpolated・occlusion_estimated・未レビューはframe全体をloss/metricsから除外する。
- **`components/augmentation.py`**: `BallDetectionAugmentation`。回転/flip/affine/crop/色/ノイズ/ゼロマスク等の augmentation 合成。

### training/
- **`lightning_module.py`**: `BallDetectionLightningModule`。Focal損失によるヒートマップ学習、GAN併用可。
- **`metrics.py`**: `BallDetectionMetrics`。ハンガリアン対応付けによる `precision`/`recall`/`f1`/`mean_distance_px`。
- **`candidate_recall.py`**: native heatmapから毎epochのvalidation候補recallを集計。source座標・単一observed教師・重複frameの窓選択を検証する。
- **`runner.py`**: `BallDetectionTrainingRunner`。datamodule/lightning_module構築と、2D detectorのweights-only初期化。3D court metadataを要求せず、module全体の重みをstrictに復元する（GAN有効時のdiscriminatorも含む）。不完全な重み転送は拒否する。

### inference/
- **`checkpoint.py`**: predictor・レビューUI共通の推論専用loader。保存されたmodel設定・`model.`重みと入力正規化をstrict復元する。`data.augmentation.normalize_imagenet.enabled`は必須で、有効なら保存されたmean/stdも使う。学習専用オプションは要求・補完しない。
- **`predictor.py`**: `BallDetectionPredictor`。checkpointのadapterを維持し、CPU上の `BallPrediction`（点・score・native heatmap・候補の局所特徴）を返す。

### evaluation/
- **`candidate_recall.py`**: 閾値なし候補集合のsource画素recall、候補外、順位誤りの加算可能な件数。[refinerのvalidation選定](../ball_refiner/README.md#validationによる検出器選定)で利用する。
- **`contracts.py`**: 評価マニフェスト(`ball_detection_evaluation_manifest_v1`)の型付き契約。
- **`configuration.py`**: checkpoint設定読み出しとモデル名整合性検証。
- **`dataset_provenance.py`**: データセットの provenance(ハッシュ・ソース)記録。
- **`metrics.py`**: `StratifiedBallMetrics`。全体/データソース別のメトリクス追跡。
- **`evaluator.py`**: 1 job(checkpoint×dataset×split) を評価する `DefaultJobEvaluator`。
- **`reporting.py`**: `summary.json`/`comparison.csv`/`comparison.md` を生成。
- **`runner.py`**: `EvaluationPipeline`。fingerprintベースの再利用付き複数job評価。
- **`holdout_inference.py` / `holdout_metrics.py`**: storeの全frameを一度ずつ数える推論と、元動画画素での欠損・誤差・camera/注釈/手首距離別集計。[Meiji比較benchmark](../../../tests/benchmarks/README.md#meiji-ball-holdout)から使う。

### visualization/
- **`orchestrator.py`**: checkpointからのスライディングウィンドウ推論→GIF保存を統括。
- **`adapters/predict_inputs.py`**: スライディングウィンドウ開始位置とバッチ構築。
- **`adapters/render_inputs.py`**: 学習バッチの正規化契約を受け、RGB表示は逆正規化し、MDD表示は学習と同じ正規化済み画像から生成する。
- **`api/predict.py`**: `predict_clip()`。重複ウィンドウ推論の集約と `PredictionSequence` 構築。
- **`io/clip.py`**: storeのclipから推論/描画用テンソルを構築（`visualization.store_dir` と `clip_id` を指定）。
- **`rendering/clip_renderer.py`**: RGB/MDD/予測/heatmapの2x2グリッド描画。
- **`review/datasets.py`**: `BallDatasetCatalog`。ball storeの全versionを走査し、シーン(opaque ID)・dense frame位置・multi-instance `FrameLabel` を提供する。
- **`review/checkpoints.py`**: `scan_checkpoints()`。checkpoint本体の保存configから `model.name`・`num_frames`・窓下限・metrics既定を読む。
- **`inference/loader.py`**: `load_ball_model()`。共通checkpoint loaderを使い、レビュー用の入力サイズ・窓長を検証する。
- **`inference/peaks.py`**: `decode_frame_peaks()`。canonicalなthreshold/NMS/top-k + subpixel refineで複数peakをstored image pixelへ写す。
- **`inference/rasters.py`**: 予測probability heatmapのRGBA overlay。
- **`inference/service.py`**: `DetectionService`。catalog/scenes/preview/image/validate/inferを提供する共有Webバックエンド。

### generate_dataset/
- **`frame_store/`**: 統一 frame store の生成。`sources/{tracknet,meiji,chat_annotation}.py` が各注釈形式を検証して `ClipSpec`(`clip.py`)へ写し、`builder.py` が split 割当・JPEG shard 化・アトミック publish を行う。設定は `configs/generate_dataset.yaml`(`config.py` で厳密検証)、入口は `scripts/generate_dataset.py`。

### scripts/
- **`review_play_intervals.py`**: CPUの区間推定・実画像レビュー。
- **`train_mdd_pose.py`**: レビュー後に使うMDD＋pose座標モデルの学習入口。epoch/学習率/seedを明示し、testを読まない。
- **`generate_dataset.py`**: 統一 frame store の生成エントリポイント。
- **`train.py`**: 固定長フレーム窓での通常学習エントリポイント。
- **`eval.py`**: 単一checkpointの詳細診断評価。
- **`evaluate_manifest.py`**: manifestベースの複数checkpoint比較評価。
- **`visualize.py`**: クリップ単位のGIF可視化生成。
- **`preview_augmentation.py` / `preview_heatmaps.py`**: augmentation / heatmap生成の確認用プレビュー。

### configs/
- モデル/データ/損失・メトリクス/学習/評価マニフェスト/可視化ごとにHydra設定を分割。
- 新モデルは全重みをランダム初期化する。ConvNeXtで明示的なFTを行う場合のみ`ckpt/`の重みを指定する。元学習runの重みを`init_weights`やeval/visualize/manifestの入力に使う場合は、`{role: artifact, path: ...}`でARTIFACTを明示する（契約は[出力規約](../OUTPUTS.md)）。

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
  ConvNeXtの最小1 frameにMDD multi-frame時の2 frame要件を
  加えた値。範囲外はpadせず422で拒否する。
- 推論窓は選択frame以降の連続frameで構築し、シーン長を超える要求は拒否する。
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
  symlinkはstoreを unavailable にして理由に残す。

### checkpoint互換

checkpoint本体の保存configだけを根拠にする(ファイル名から推論しない)。

- heatmap推論の `model.name` が `conv_next_unet` 以外、または
  `num_frames < アーキテクチャ最小` のcheckpointは `error` 付きで一覧に出し、
  実行時に明示的に失敗させる。
- `model.input_mode` か入力 `image_size` が欠落したcheckpointも同じく
  `error` 付きで早期に unusable とする。
- metricsは**キー欠落**なら `configs/metrics/default.yaml` の既定値を使って
  warningに残し、**値が不正**(範囲外・`nan`/`inf`・型違い)なら
  checkpointを unusable にする(黙って別のしきい値へ置き換えない)。
- dataset互換性は学習時の `data.source` ではなくアーキテクチャ制約で決める。
  storeのclipは最小窓以上の場合だけ互換とする。`scenes(checkpoint=)` はdataset単位の互換性に
  加えて**sceneごとのframe数**で絞り、短すぎるclipを実行候補に出さない。

## 学習データ

学習・評価・レビューUIは `data/ball_detection/<version>` の統一frame storeを使う。
通常学習の既定versionは `ball-mix-v2`。store内の `tracknet` / `meiji` /
`chat_annotation` は出自の名前であり、学習時に元の画像・動画へアクセスしない。
3 sourceの混合比と教師方針は `configs/data/rgb_sequence.yaml` を正本とする。
学習窓は `model.num_frames` に固定し、`eval_stride: null` はその長さごとの窓を意味する。
`data.source=store` のみ受け付け、廃止したWeb・staged・旧sourceへのフォールバックは行わない。

統一storeの新規生成は `scripts/generate_dataset.py` が担当する。TrackNetの配布形式、
Meiji、chat annotationの入力位置とsplitは `configs/generate_dataset.yaml` に定義する。
生成済みstoreの利用には原本不要だが、新規生成には選択したsourceの原本が必要。
入力やsplitを変更して生成するときは `dataset.version` に新しいversionを指定する。
生成先には全frameのJPEG shard、注釈index、metadata、READMEを保存する。
旧YouTube収集・疑似ラベル・SSL画像収集とWeb変換の入口は提供しない。

ConvNeXt checkpointの推論は保存済みmodel/正規化契約のまま利用できる。
学習再開には現在のstore設定を明示する。

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

### 毎epochの候補recallとcheckpoint

ConvNeXtの通常・GAN学習は共通の `ValidationCandidateRecall` を使う。
設定の正本は `configs/training/_validation_candidates.yaml`。
loss用に補間する前のnative sigmoid heatmapから、閾値なしK=8/NMS=5/patch=5/subpixelで復号し、
距離≤20 source pxのrecallを記録する。保存画像の端点座標をstoreのwidth-ratio scaleで割って
元動画画素へ戻す。F1のconfidence閾値・NMS・保存画像の4 px条件はこの指標に使わない。

教師はtrainingのsupervision設定と独立した、レビュー済みの単一observed球だけ。
複数instance（observed＋out_of_frameも含む）、推定位置、未レビュー、不在は分母に入れない。
Webではdistractorを除く単一可視targetを使い、保存画像をsourceとする。
datasetは未増強の教師・frame行ID・source scaleを必ずbatchへ渡し、欠落時には停止する。
検証は正規化以外の画像増強を使わない。

同じframeを複数の窓・static反復・distributed samplerで読んでも、中心への距離が最小、
同点なら早い開始位置・早い時刻の出力を1回だけ数える。DDPもframe recordを集めてから
重複を除き、epoch全体のhit/observed件数を割る。batch平均・rank平均ではない。
通常storeのvalidationはラベルによらない窓と実frameの末尾backfillで全frameを覆い、
短いclip・窓間の隙間は拒否する。混合FTのstrideは選定比較と同じ4。
評価対象はloaderが供給した一意frame。
これらの窓集合やvalidation精度設定が違うrunを、全frame/float32の選定比較と同一条件と扱わない。

`val/candidate_recall_at_8_20px` に全source合算を、
`val/<source>/candidate_recall_at_8_20px` にsource別を記録する。
camera IDのあるsourceには `val/<source>/<camera>/...` も残す。
recall@1、候補外率、順位誤り率とstrict-score版、分母・hit件数も同じprefixで記録する。
分母0のgroupには件数だけを出し、recallを0で補わない。通常のvalidation epoch全体に
observedがなければ停止する（sanity checkでは未定義の率を記録しない）。

全学習profileは `save_top_k=-1` で各epochを保持し、独立した `last.ckpt` も更新する。
候補recallをmonitorし、混合FTではMeijiだけをmonitorする。
毎epochの検証を省く設定・checkpoint無効化・有限top-k・epochを含まないfilenameは起動時に拒否する。
checkpoint削除を伴うqueueの `--prune-ckpt` は使わない。
既存checkpointの推論契約はそのままで、今後の学習にはこの明示設定を使う。

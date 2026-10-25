# Ball Detection

RGB＋MDDから2Dボール位置を推定する。任意長T・5chのheatmap構成と既存32フレーム構成の
入力仕様は [models/mdd](models/mdd/README.md) を参照。
既存ConvNeXtのnative heatmap契約と、MDDの座標回帰契約を明示的に分ける。

モデル事前推論から注釈修正・増分公開・累積追加学習への変更後設計は、
[注釈・継続学習のアーキテクチャ](../base/annotation/README.md)を参照する。
実装状況は同資料からリンクしたサブIssueで管理する。

## 公開推論重み

sceneで使用するConvNeXtの推論用checkpointは
[Hugging Face](https://huggingface.co/Motoki098/tennis-lab-models)から取得できます。repo rootで実行してください。

```bash
.venv/bin/python -m src.utils.model_weights --checkpoint-root "$PWD/ckpt" --model ball
```

SHA検査・既存ファイルの扱い・再現手順は[重み取得CLI](../../utils/README.md#published-detector-weights)、
データの出自とCC BY-NC 4.0の利用条件は
[LICENSES.ja.md](../../../LICENSES.ja.md#published-artifacts)を参照してください。
配置先は標準pipelineと同じ`ckpt/ball_detection/i935-mixed-ft-s42-epoch09.ckpt`です。
公開版は学習状態を除去しており、推論用です。

## 構成と責任

| 場所 | 責任 |
|---|---|
| `configs/` | モデル・データ・augmentation・既存Hydraレシピの設定 |
| [data/](data/README.md) | frame storeと注釈の正本、教師mask、RGB/GTの同期変換 |
| [data/windows/](data/windows/README.md) | プレイ区間、凍結manifest、32frame/FPS窓、poseの有無とidentity検証 |
| [models/mdd/](models/mdd/README.md) | 圧縮・pose・CNNの比較候補とモデル構成。学習・データ探索・queue制御を持たない |
| `models/conv_next_unet.py` | sceneで使用する既存heatmapモデル |
| `preprocessing/mdd.py` | モデルが使う固定FP32 MDDの唯一の実装 |
| `model_io/` | 入出力の検証、モデル生成、座標/heatmapの別々の契約 |
| `checkpoints/` | schema別の保存・読込・encoder転送・再現性の照合 |
| [training/](training/README.md) | 等予算の可変長heatmap、5chのDPT→query→hybridと継続学習、既存heatmap/座標学習、共通runtime |
| `evaluation/` | 学習中/単独評価で共通の採点、重複排除、clean/stress |
| `inference/` | scene向け公開predictorとtrajectory gate |
| [annotation/](annotation/README.md) | 新規clipの事前推論・注釈レビュー・累積教師公開 |
| [visualization/](visualization/README.md) | WebUI・描画・GIF・教師/拡張preview |
| [dataset_statistics/](dataset_statistics/README.md) | 注釈構成・欠損・位置・速度・窓・pose品質のCPU統計 |
| `generate_dataset/` | 原本から統一frame storeを生成 |
| `scripts/` | 下表の操作入口。学習loop・queue制御を実装しない |

## 公開コマンド

`python`はrepoの`.venv/bin/python`を使う。GPU処理は共有training queue経由で実行する。

| module末尾（`src.tasks.ball_detection.scripts.`） | 選択と用途 |
|---|---|
| `generate_dataset` | 設定からframe storeを生成 |
| `prepare_data` | 凍結したプレイ窓・pose/MDD-onlyのmanifestと比較設定を準備 |
| `train` | `--stage convnext_heatmap\|dpt_pretrain\|coordinates\|hybrid\|temporal`。新heatmapはarchitectureとphaseも明示 |
| `evaluate` | `--mode heatmap\|manifest\|coordinates\|hybrid\|temporal`。保存されたcheckpoint契約で採点 |
| `review` | `--mode dataset\|inference`でWebUIを起動 |
| `visualize` | checkpointのclip推論をGIFへ描画（overlayはMP4も出力） |
| `visualize_temporal_heatmaps` | 同じclip区間のowner予測を複数のT/アーキテクチャで横に並べてMP4へ出力 |
| `visualize_annotation_correction` | 既存のモデル候補とCodex確定注釈をrank／左右比較。手順は[可視化ガイド](visualization/README.md#モデル候補とcodex確定注釈の発表用比較) |
| `preview` | `--kind spatial\|heatmaps\|video`で教師・画像変換を確認 |

BLCS損失比較の入口もこの`scripts/`配下に集約する。`train_blcs_ablation`は明示した条件・seed・更新数で学習し、`evaluate_blcs_ablation`はcheckpoint／対照を保存評価する。実装と仕様は[BLCS ablation](../blcs/ablation/README.md)に置く。

選択後に`--help`を付けると、その操作の引数を表示する。未対応のstage/schema/inputを推測して読み替えない。
学習レシピ・再開条件は[training README](training/README.md)、窓とFPSの仕様は[data/windows](data/windows/README.md)が正本。

モデル比較の反復起動・特定runの復旧・2時間監視は既存queue/follow-up skillと実行記録が担当する。
applicationにcampaign専用controllerや固定run IDを置かない。
汎用性のある速度・復号・数値検証は[benchmarks](../../../tests/benchmarks/ball_detection/README.md)から実行する。
過去の試行は`knowledge/`に記録されたcommitで再現し、実行状態・生ログ・checkpoint・大量のGIFは`outputs/ball_detection/`へ保存する。

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

### ハイブリッドの出力契約

5ch query＋DPTは [`model_io/mdd_hybrid.py`](model_io/mdd_hybrid.py) の
`build_mdd_hybrid_detector` でモデルとadapterを構築する。
学習済みモデルは同moduleの`load_mdd_hybrid_detector(path)`で、保存された存在閾値とともに復元する。
`MDDQueryInput`（RGB uint8と実timestamps）を渡し、`BallPrediction`を継承した
`BallHybridPrediction` を得る。構造・重み転送は [models/mdd](models/mdd/README.md) を参照。

`coords` はnative heatmapのargmax＋任意のlog-parabolic subpixel補正で、
`x/(W_heatmap−1), y/(H_heatmap−1)`。RGBへ戻すときはRGBの `(W−1,H−1)` を掛ける。
`presence_logits` と `present`（bool）を追加し、存在確率の閾値 `presence_threshold` は
呼出側の必須指定とする。`present` はqueryのsigmoid値がこの閾値以上かで決める。
閾値の較正・heatmapとの複合判定は学習・評価側の判断事項。

不在frameの `coords` はNaN、`confidence` は0、全候補の `valid` / `patch_valid` はfalse、
候補値は0。下流は必ず `present` で座標を選別し、JSON等へ出す際は不在をnull等で表す。
`heatmaps` は存在gateを掛けないnative sigmoid値を保持し、存在logitと別々に監査できる。
全fieldはCPU tensor。既存の座標だけを扱う `predict_coordinates` はこのモデルを拒否する。
既存scene用ConvNeXt factory／UIへ暗黙に接続せず、hybrid用adapterを明示的に使う。

## データセットレビュー / 推論UI

起動コマンドと操作手順は[Web UI利用ガイド](visualization/README.md)、HTTP APIは
[共有Detection UI](../base/visualization/detection/README.md)を参照してください。ここには
ball_detection固有のsourceと互換契約だけを記す。

### source

| id | 実体 | mode | 備考 |
|---|---|---|---|
| `store/<version>` | `data/ball_detection/<version>` | temporal | TrackNet・Meiji・chat_annotation。1 camera-clip = 1 scene |
| `pose-approved/<name>` | `--play-poses`で指定したpose datasetのball snapshot | temporal | pose承認済みclipだけ。起動方法は[利用ガイド](visualization/README.md#プレイ区間の候補) |

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
- pose reviewは明示指定したdata root内のpose manifestを読み、その参照先をproject root内の
  snapshotに限定する。metadata/indexのhash、clip集合の一致とshardのstore内配置を検証し、
  live storeへ置き換えない。破損時は理由付きでdatasetを無効にする。

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

現行のdataset系列・sourceとsplit・入力と教師・派生poseの制限は
[データセット体系](data/README.md)を参照してください。
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
.venv/bin/python -m src.tasks.ball_detection.scripts.train --stage convnext_heatmap --config-name train_meiji_mixed \
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

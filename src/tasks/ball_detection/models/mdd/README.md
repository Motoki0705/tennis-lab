# MDD model families

`config.py`が構成と有効な組合せを検証し、`detectors.py`がencoderとheadを組み立てる。学習段階・queue・datasetの責任を持たない。

| 部品 | 役割 |
|---|---|
| `token_encoder.py` | Conv2d / Average / Pixel Unshuffle / Haarの1/16圧縮と、局所2D/3D処理による1/64 token化 |
| `cnn_encoder.py`、`blocks.py` | 深い残差/ConvNeXt V2系/FasterNet系CNN。4解像度の特徴を返し、dense3d/factorizedの時間処理を選択 |
| `pose.py` | DeepSets / Attention Pooling / Hierarchical Transformer / Skeleton GNNによるframeごとのpose token |
| `attention.py` | 同時刻のCross-Attentionと実時間RoPEのSelf-Attention。pose融合とquery-onlyの違いは明示する |
| `detectors.py` | MDDPoseDetector、MDDQueryDetector、MDDDPTDetector、DeepMDDQueryDetector |
| `hybrid.py` | DeepMDDHybridDetector。時間queryを最下層にconcatし、DPT heatmapと存在logitを返す |
| `variants.py` | 有効なpose32条件＋poseなしquery-only4条件を列挙 |

圧縮比較用encoderと深いCNNは異なる構造として維持する。深いCNN候補を圧縮方式と無条件に直積しない。
同じquery attentionを共有し、比較軸が増えても専用の学習CLIを増やさない。

深いCNNの新規5ch構成は `configs/model/mdd_dpt_rgb_mdd.yaml` で選ぶ。
入力契約の正本は [`preprocessing/rgb_mdd.py`](../../preprocessing/rgb_mdd.py) の
`RGBToRGBMDD.input_contract()`。外部入力は RGB uint8 `(B,32,3,H,W)`、CNN入力は
FP32 `(B,5,32,H,W)`、順序は `R,G,B,MDD_positive,MDD_negative`。
RGBは255で割った `[0,1]` のみで、ImageNet正規化を加えない。MDDも `[0,1]` で、
既存 `RGBToMDD` のFP32計算・係数・演算順をそのまま使う。
サンプル済みの同じ拡張後RGBから5chを作るため、GTと空間・時刻が一致する。
窓先頭はMDDの2chだけゼロで、RGBの3chは実画像を保持する。

`MDDPretrainConfig.input_channels=5` が5chを明示する。旧設定の省略値は2chを意味し、
旧MDD-only DPT/queryのstate_dict名・計算順・入力契約を維持する。
保存した `recipe.input_contract` と設定のチャンネル数を読込・encoder転送時に照合し、
2ch/5ch間の転送や正規化・順序の読み替えを拒否する。
RGB特徴の並列branchはない。圧縮比較・pose系は従来の2ch契約のまま。
mask/実PTSはmetadataであり、pose不要モデルはpose artifactを開かない。court入力なし。
query-onlyは同時刻MDDへの片方向Cross→query列の時間Self→FFN。poseありの双方向融合は別構造。
深いquery decoderはSwiGLU（中間幅8/3 dimを64単位で切上げ）、dim256なら704。

DPTは`src/utils/models/dpt.py`をcourtと共有し、CNNの4特徴を融合して入力1/4のheatmap logitsを出す。
2段目の事後学習は同一CNNだけをstrict転送し、新規query decoderを学習する。encoder固定/解除は学習側が制御する。

## Query＋DPTハイブリッド

`MDDHybridConfig` / `configs/model/mdd_hybrid.yaml` は5chを必須とする。
`DeepMDDQueryTrunk` が既存のencoderとquery経路を共有する。32フレームに対する
時間Self-Attention後のquery `(B,T,dim)` を、1/64特徴の各セルへ同じ値でbroadcastし、
チャンネル方向にconcatする。1/8・1/16・1/32特徴はそのままDPTに渡す。
DPTの最下層入力幅だけ `mixed_channels[-1] + dim` に拡張し、既存の1×1投影で
`decoder_channels` に揃える。別の圧縮層を加えず、concatの情報を直接投影できる。

`forward(rgb, timestamps)` は `MDDHybridOutput` を返す。

| field | shape | 用途 |
|---|---|---|
| `heatmap_logits` | `(B,32,ceil(H/4),ceil(W/4))` | native格子のdense教師。不在では全面0を教師にできる |
| `presence_logits` | `(B,32)` | queryのLayerNorm→Linearから得る存在logit |

forwardは存在判定でheatmapをmaskしないため、両方の損失を独立に学習できる。
座標回帰headは持たない。推論用の座標・不在契約は
[ball_detectionの出力契約](../../README.md#ハイブリッドの出力契約)を参照。

## 3段目への重み転送

[`initialize_hybrid_from_query`](../../checkpoints/hybrid.py) は2段目のquery checkpointと
`MDDHybridConfig` から新規モデルと転送receiptを返す。

| 処理 | 登録名 |
|---|---|
| 2段目から転送 | `encoder.*`, `projection.*`, `spatial_position.*`, `ball_query`, `blocks.*`（RoPE bufferも含む） |
| 新規初期化 | `decoder.*`, `heatmap_head.*`, `presence_head.*` |
| 転送対象外 | 2段目の座標回帰 `head.*`、optimizer、freeze状態 |

1段目checkpointへの参照は来歴としてのみ扱い、ファイルを開かず、DPT重みを使わない。
入力契約と転送部の構成を照合し、余分・欠落・形状違いのstateを拒否する。
DPT幅とactivation checkpointingだけは転送部の重みに影響しないため変更可能。
receiptには読み込んだcheckpointのhash、転送・新規初期化の全stateキー、構成を残す。
返却モデルの全parameterは学習可能。freeze、損失配分、保存・再開レシピは学習側が決める。

3段階の実行・必須Hydra設定・不在教師・来歴・resume/継続学習の違いは
[training README](../../training/README.md#学習済み重みと再開)を参照。
学習済みhybridの保存schemaは`mdd_hybrid_training.v1`。
`checkpoints/hybrid_training.py`の`restore_hybrid_model`で、親のcheckpointを開かずに復元する。
新snapshotの継続学習では、3段目からDPTと存在headも含む全重みを引き継ぐ。
query→hybridでDPTを新規初期化する処理とは別の初期化modeである。

ファイル配置の整理でstate_dictの登録名・forwardの計算順は変更しない。
既存のscene用ConvNeXtは`../conv_next_unet.py`のnative heatmap・候補・patch契約を維持する。
座標出力からheatmapを捏造せず、scene/refinerへの座標モデル接続は別の契約として扱う。

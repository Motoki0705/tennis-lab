# MDD model families

`config.py`が構成と有効な組合せを検証し、`detectors.py`がencoderとheadを組み立てる。学習段階・queue・datasetの責任を持たない。

| 部品 | 役割 |
|---|---|
| `token_encoder.py` | Conv2d / Average / Pixel Unshuffle / Haarの1/16圧縮と、局所2D/3D処理による1/64 token化 |
| `cnn_encoder.py`、`blocks.py` | 深い残差/ConvNeXt V2系/FasterNet系CNN。4解像度の特徴を返し、dense3d/factorizedの時間処理を選択 |
| `pose.py` | DeepSets / Attention Pooling / Hierarchical Transformer / Skeleton GNNによるframeごとのpose token |
| `attention.py` | 同時刻のCross-Attentionと実時間RoPEのSelf-Attention。pose融合とquery-onlyの違いは明示する |
| `detectors.py` | MDDPoseDetector、MDDQueryDetector、MDDDPTDetector、DeepMDDQueryDetector |
| `hybrid.py` | 時間queryを最下層にconcat。heatmap専用と、既存の存在head付きhybridを別classで構築 |
| `execution.py` | frame分割とactivation checkpointing。学習parameterや時間長依存のstateを持たない |
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

`MDDPretrainConfig.input_channels` は必須で、5chは`5`、MDD-only比較構成は`2`を明示する。
Python構築・YAML設定での省略は拒否する。checkpointの履歴形式は
[`checkpoints`](../../checkpoints/README.md) の明示的な識別条件に従う。
MDD-only DPT/queryのstate_dict名・計算順は維持する。
保存した `recipe.input_contract` と設定のチャンネル数を読込・encoder転送時に照合し、
2ch/5ch間の転送や正規化・順序の読み替えを拒否する。
RGB特徴の並列branchはない。圧縮比較・pose系は従来の2ch契約のまま。
mask/実PTSはmetadataであり、pose不要モデルはpose artifactを開かない。court入力なし。
query-onlyは同時刻MDDへの片方向Cross→query列の時間Self→FFN。poseありの双方向融合は別構造。
RGBのdtype・shape・設定したフレーム数は`model_io`で検証する。DPTの学習・評価もこの境界を通し、
5ch変換・CNNのforwardには検証分岐を置かない。CNNの入力チャンネル数は設定と固定変換で決まる。
深いquery decoderはSwiGLU（中間幅8/3 dimを64単位で切上げ）、dim256なら704。

DPTは`src/utils/models/dpt.py`をcourtと共有し、CNNの4特徴を融合して入力1/4のheatmap logitsを出す。
2段目の事後学習は同一CNNだけをstrict転送し、新規query decoderを学習する。encoder固定/解除は学習側が制御する。

## 任意長のheatmap専用モデル

`MDDHeatmapConfig` / `configs/model/mdd_dpt_heatmap.yaml` はDPT、
`MDDHeatmapHybridConfig` / `configs/model/mdd_heatmap_hybrid.yaml` は
`DeepMDDHeatmapHybridDetector` を選ぶ。両方とも5ch入力とnative /4 logits
`(B,T,ceil(H/4),ceil(W/4))` を持ち、存在head・存在loss・存在gateを持たない。
720pの入力をresizeせずに渡すと、出力は180×320になる。
CNNはConvNeXt V2系factorized、DPT幅128、queryは幅256・4層・8 heads・FFN704を維持する。
Tを変えても、チャンネル数や層数は変えない。

| 設定 | 意味 |
|---|---|
| `frames` | 保存する入力長T。1以上の整数。MDDの最初のframeは0、T=1も有効 |
| `frame_chunk_size` | nullで分割無効、正整数でCNNとframe-local DPTを時間分割。既定null |
| `activation_checkpointing` | CNNの各spatial stage、queryの各block、DPTからheatmap headまでの各frame chunkを再計算。既定false |

CNNは2層の時間方向kernel=3、stride=1で、特徴の時間半径は順に0/0/1/2。
分割時は前後2枚の実frameをhaloとして読み、4特徴すべてからhaloを除いて結合する。
実clip端だけでzero paddingを行う。RGB→MDDは分割前に実行するため、chunk先頭で差分を0へ戻さない。
GroupNorm・GRNはframeごとで、halo外への時間依存を作らない。
DPTはhaloなしで分割し、hybridのqueryは分割せず全Tの実PTSをRoPEへ渡す。
checkpointingは学習中かつ勾配有効時に適用し、dropoutのRNGを保存する。
CNNは分割なしでもhalo付きchunk内でもspatial stageごとにcheckpointする。
stage境界の入力は保持するが、backwardでencoder全体の中間activationを一度に再生成せず、
再計算時のactivationを1 stage分に区切る。低解像度の/32・/64にある時間混合層は従来どおりcheckpoint対象外。
DPTはframe-localなのでchunk全体を1区間にし、chunkの4特徴入力だけを保持して、
広い/4特徴を含むdecoder・headの中間activationを再計算する。
分割なしではDPTの区間は全Tとなるため、長いTで再計算時のpeakも抑える場合はframe_chunk_sizeを明示する。
queryはblock入力のquery・共有patch・PTSを保持し、attention・FFNの中間activationをblockごとに再計算する。
分割だけではbackward用activationの保持は減らないため、学習時の保持削減にはcheckpointingも使う。
前処理のFP32 5ch入力・4段のCNN特徴・全Tのqueryはメモリに保持する。720pの実行上限は別途実測する。

入力検証・推論は `model_io/mdd_heatmap.py` の `build_mdd_heatmap_detector(config)` を使う。
両モデルとも `MDDQueryInput(rgb, timestamps)` を受け取り、学習にはadapterの
`build_call`から得た引数で生logitsを使う。DPTはtimestampsを検証し、forwardにはRGBだけを渡す。
推論は `BallPrediction` のnative sigmoid heatmap・座標・候補を返す。弱い出力も存在判定で除外しない。
教師は [共通heatmap objective](../../training/README.md#共通heatmap-objective) が担当する。

`heatmap_model_payload(model)` / `restore_heatmap_model(payload)` は重み専用artifact
`schema=mdd_heatmap_model.v1` を保存・復元する。model kindはDPTが `mdd_dpt_heatmap`、
hybridが `mdd_heatmap_hybrid`。`model_config`の全項目（framesとframe_chunk_sizeを含む）、
`input_contract`、`state_dict`を持つ。設定のTと入力契約のTの不一致・欠落は拒否する。
ファイルから推論する場合は `load_mdd_heatmap_detector(path)` を使う。
optimizer・学習予算・resume状態の保存は学習側の責任である。

同じ構成のstate_dictはT=32/64/128/256へ補間なしでstrict loadできる。
Conv3dのkernel、空間座標投影、query、RoPEの周波数bufferはTに依存しない。
系列長を変えるfine-tuningは、保存モデルを元のTで検証してから、明示した新しいTのモデルへ
全stateをstrict転送する。artifactのTを書き換えて読込やresumeに見せない。
既存 `MDDPretrainConfig` と存在head付き `MDDHybridConfig` は従来の32frame保存形式のまま。
新artifactへ暗黙に変換せず、存在head付きstateをheatmap専用モデルへ読むとstrict loadで拒否する。

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

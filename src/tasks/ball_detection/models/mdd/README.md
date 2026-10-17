# MDD model families

`config.py`が構成と有効な組合せを検証し、`detectors.py`がencoderとheadを組み立てる。学習段階・queue・datasetの責任を持たない。

| 部品 | 役割 |
|---|---|
| `token_encoder.py` | Conv2d / Average / Pixel Unshuffle / Haarの1/16圧縮と、局所2D/3D処理による1/64 token化 |
| `cnn_encoder.py`、`blocks.py` | 深い残差/ConvNeXt V2系/FasterNet系CNN。4解像度の特徴を返し、dense3d/factorizedの時間処理を選択 |
| `pose.py` | DeepSets / Attention Pooling / Hierarchical Transformer / Skeleton GNNによるframeごとのpose token |
| `attention.py` | 同時刻のCross-Attentionと実時間RoPEのSelf-Attention。pose融合とquery-onlyの違いは明示する |
| `detectors.py` | MDDPoseDetector、MDDQueryDetector、MDDDPTDetector、DeepMDDQueryDetector |
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
RGBのdtype・shape・32フレーム契約は`model_io`で検証する。DPTの学習・評価もこの境界を通し、
5ch変換・CNNのforwardには検証分岐を置かない。CNNの入力チャンネル数は設定と固定変換で決まる。
深いquery decoderはSwiGLU（中間幅8/3 dimを64単位で切上げ）、dim256なら704。

DPTは`src/utils/models/dpt.py`をcourtと共有し、CNNの4特徴を融合して入力1/4のheatmap logitsを出す。
事後学習は同一CNNだけをstrict転送し、新規query decoderを学習する。encoder固定/解除は学習側が制御する。

ファイル配置の整理でstate_dictの登録名・forwardの計算順は変更しない。
既存のscene用ConvNeXtは`../conv_next_unet.py`のnative heatmap・候補・patch契約を維持する。
座標出力からheatmapを捏造せず、scene/refinerへの座標モデル接続は別の契約として扱う。

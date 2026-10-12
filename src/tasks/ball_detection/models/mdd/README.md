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

座標モデルはRGB uint8 `(B,32,3,H,W)`を受け、内部で固定FP32 MDDを計算する。RGB特徴の並列branchはない。
mask/実PTSはmetadataであり、pose不要モデルはpose artifactを開かない。court入力なし。
query-onlyは同時刻MDDへの片方向Cross→query列の時間Self→FFN。poseありの双方向融合は別構造。
深いquery decoderはSwiGLU（中間幅8/3 dimを64単位で切上げ）、dim256なら704。

DPTは`src/utils/models/dpt.py`をcourtと共有し、CNNの4特徴を融合して入力1/4のheatmap logitsを出す。
事後学習は同一CNNだけをstrict転送し、新規query decoderを学習する。encoder固定/解除は学習側が制御する。

ファイル配置の整理でstate_dictの登録名・forwardの計算順は変更しない。
既存のscene用ConvNeXtは`../conv_next_unet.py`のnative heatmap・候補・patch契約を維持する。
座標出力からheatmapを捏造せず、scene/refinerへの座標モデル接続は別の契約として扱う。

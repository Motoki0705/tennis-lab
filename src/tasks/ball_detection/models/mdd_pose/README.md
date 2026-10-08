# MDD coordinate detector（poseあり32条件＋query-only 4条件）

外部入力はRGB順uint8のclip `B,T,3,H,W`。モデル先頭の固定`RGBToMDD` layerがFP32で2ch MDDを作る。
poseありの32条件はCOCO17の2D座標も使う。
query-onlyの4条件にはpose入力・pose module・null pose tokenを設けない。
学習部が使う画像特徴はMDDだけ。RGBから学習部への別経路はない。出力は32枚それぞれのsource正規化座標 `(B,32,2)`。
公開境界は`model_io.mdd_pose.build_mdd_pose_detector`のmodel/adapter pairで、
`MDDPoseInput.rgb`・pose・時刻を検証してから、MDD生成を含むforwardを実行する。
query-onlyの公開境界は`model_io.mdd_query.build_mdd_query_detector`と`MDDQueryInput`。
設定の正本は `../../configs/model/mdd_pose.yaml` と `../../configs/model/mdd_query.yaml`。
全重みをランダム初期化する。
refinerの候補/patch契約との接続は#986の未決事項で、座標からheatmapを捏造しない。

固定前処理の定義・色順序は[共通前処理](../../preprocessing/README.md)を正本とする。
RGB→MDDの計算はBF16 autocast下でもFP32を保ち、係数は学習しない。
`torch.compile`にはこの前処理から座標headまでを含める。

## Encoder: frame独立の1/16圧縮 → 共通2D/2D/3Dを2回 → 1/64

比較するのは**最初の空間1/16への圧縮だけ**。ここでは時間を混ぜない。
全方式の出力channelを`stem_channels[-1]`へ揃える。

| 方式 | 高解像度MDDから1/16への処理 |
|---|---|
| `conv2d` | frameごとのConv2d、kernel=3×3・stride=2を4段 |
| `average` | 16×16 Average Pooling → 1×1 Conv2d |
| `unshuffle` | PixelUnshuffle(16)、2→512ch → 1×1 Conv2d |
| `haar` | 全帯域を再帰分解する4段Haar wavelet packet、2→512ch → 1×1 Conv2d |

DWTはLLだけの多段分解ではなく、全subbandを同じ1/16格子へ保持する。
Unshuffle/DWTの変換自体は可逆だが、その後のchannel投影まで可逆とはしない。
Conv2d方式は各段、他の3方式はchannel投影後にchannel LayerNorm＋GELUを置く。

以降は全方式で同じ **`2D → 2D → 3D`** blockを2回使う。

| block内の順序 | kernel | stride (T,H,W) | 役割 |
|---|---|---|---|
| 1. frameごとのConv2d | 3×3 | (1,2,2) | 空間を1/2へ |
| 2. frameごとのConv2d | 3×3 | (1,1,1) | 同じframe内の空間処理 |
| 3. Conv3d | 3×3×3 | (1,1,1) | 前後1frameとの局所混合 |

各層はpadding=1、channel LayerNorm→GELUを続ける。
正規化は時空間位置ごとのchannel方向だけで、時間を跨ぐBatch/GroupNormは使わない。
時間strideは全段1。3D層は各々t−1,t,t＋1だけを参照し、2層の合成で
encoder全体のMDD受容野は**5frame（t±2）**となる。MDD生成のRGB差分参照範囲は別。
窓端はfeatureのゼロpaddingで、入力RGB frameを反復しない。

既定の720×1280・32frame入力:

| 段 | C×T×H×W |
|---|---|
| MDD | 2×32×720×1280 |
| 1/16 stem | 48×32×45×80 |
| 共通block 1: 2D→2D→3D | 64×32×23×40 |
| 共通block 2: 2D→2D→3D | 96×32×12×20 |
| 1×1×1投影＋格子XY埋込 | 128×32×12×20 |
| frameごとのtoken列 | **32×240×128** |

学習可能stemの途中channelは`8→16→32→48`。
MDDをresize/cropせず、1/16の変換に必要な場合だけ下/右を16の倍数までゼロpaddingする。
720×1280はすでに16の倍数。後段Conv2dの通常paddingにより45→23→12となり、
最終token数は`ceil(H/64) × ceil(W/64)`。各tokenは実画像と重なり、端の部分cellの
位置埋込は実画像内の範囲の中心を使う。pose/教師座標の正規化にはpadding後サイズを使わない。
最終patch tensorはfloat32約3.75MiB。高解像度stem等のactivation memoryは別途必要。

## poseありの集約・融合・座標

`pooling.py` は各frameの全人物を1 pose tokenへ集約する。
座標をsourceのW−1/H−1で正規化し、confidence/ID値は特徴として入力しない。
人物軸は可変で、clip内の固定player列を維持するが並べ替えにも不変。
欠測はmaskで除外し、人物0人/all-maskedには学習可能なnull tokenを使う。

- `deepsets`: 人物の17×2座標をMLP→人物間masked mean→MLP。
- `attention`: 人物MLP→1つのpose queryによるattention pooling。
- `hierarchical`: 関節埋込＋関節ID→人物内attention/pooling→人物間attention/pooling。
- `gnn`: COCO骨格edge＋self-loop、2段の正規化GCN→関節mean→人物attention pooling。

`model.py` の各融合blockは以下の順序で処理する。

1. **同一frameだけのCross-Attention**。MDD patchが同時刻のpose/queryを参照して更新され、
   pose/queryも同時刻の更新済みMDD patchを参照する。frame軸をbatchへ畳むため別時刻のkeyを読まない。
2. **pose／ball query列だけの時間Self-Attention**。実PTS秒のRoPEを使い、32frameをoffline双方向に混ぜる。
3. pose/query側のFFN。

この融合blockを既定2回繰り返す。MDD patch列へ時間Self-Attentionや大域空間Self-Attentionは適用しない。
後続blockでは既に時間混合されたpose/queryを同時刻のcross-attentionで参照できるため、
モデル全体の参照範囲はencoderの±2frameに限定されない。

`query`条件はframeごとのball query、`pose`条件はpose tokenをreadoutにし、
LayerNorm→Linear(2)→sigmoidでuvを出す。pooling内部のpose queryはball queryとは別。
4種類の1/16圧縮×4pose集約×2readout＝32条件を同じ実装で選べる。

## poseなしのquery-only

`readout: query_only`と`pose_pooling: null`を一緒に指定し、`MDDQueryDetector`を使う。
4つの圧縮方式を比較し、pose集約との積は取らない。全体は32＋4＝**36条件**。

1. 各frameのball queryが**同時刻のMDD patchだけ**をCross-Attentionで読む。
2. query列の32tokenに、実PTS秒の時間Self-Attentionを適用する。
3. query側FFNを通す。各更新は残差加算で、既定2 block。

MDD tokenをqueryから更新する経路はなく、同じencoder出力を各blockから参照する。
球queryの初期ベクトルをframe間で共有し、最後に各時刻のqueryからuvを回帰する。
RGB uint8＋実時刻を受け取るtyped adapterで検証し、poseあり設定や旧MDD tensorとの取り違えを拒否する。

## データ・混合FPS・学習入口

元FPS/1/2/1/4を同runで混ぜ、各条件で32枚を保つ。MDDは間引いたRGBから再計算し、
pose・教師・mask・実PTSも同じframeを読む。40%のpose生成条件からquery-onlyの学習選択を独立させる。
固定manifestの準備、36構成、全体／共通subsetの評価と実行手順は
[座標モデルの学習準備](../../training/COORDINATE_TRAINING.md)を参照。

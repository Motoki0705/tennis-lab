# MDD＋pose coordinate detector（学習前レビュー用）

入力は高解像度の2ch MDDとCOCO17の2D座標のみ。RGBはMDD計算の素材で、
モデルへ直接渡さない。出力は32実frameそれぞれのsource正規化座標 `(B,32,2)`。
公開境界は`model_io.mdd_pose.build_mdd_pose_detector`のmodel/adapter pairで、
`MDDPoseInput`を検証してから計算のみのforwardを実行する。
設定の正本は `../../configs/model/mdd_pose.yaml`。全重みをランダム初期化する。
refinerの候補/patch契約との接続は#986の未決事項で、座標からheatmapを捏造しない。

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

## pose・融合・座標

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

## データと学習入口

先に `scripts.review_play_intervals` でapproved pose clipと実frameの学習窓を固定し、
可視化をレビューする。manifestにコピーされたpose artifact hashを使うため、
元campaignの承認が進んでも対象は自動で増えない。
MDDは保存JPEGの解像度で計算し、ImageNet正規化や先行resizeはしない。
ConvNeXtと共有するsigmoid MDDを使い、最初のframeは参照画像がないため0にする。
位置教師は単一observedだけ。欠損を補間した座標で学習しない。

レビュー・予算確定後の入口（今回の作業では実行しない）:

```bash
# CUDAは元repo共有training queueから実行する。
.venv/bin/python -m src.tasks.ball_detection.scripts.train_mdd_pose \
  --manifest <absolute-play-review>/manifest.json --output <absolute-new-run> \
  --device cuda --epochs <budget> --learning-rate <lr> --seed <seed> \
  --compression conv2d --pose-pooling attention --readout query
```

初期実装のlossはobserved uvのSmoothL1、val選択は重複frameを中心窓規則で
一度ずつ数えたsource画素の平均誤差。testはtrainerから読まない。
これらの設定も学習前レビューの対象であり、まだ実データ学習・性能評価は行っていない。

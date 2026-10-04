# MDD＋pose coordinate detector（学習前レビュー用）

入力は高解像度の2ch MDDとCOCO17の2D座標のみ。RGBはMDD計算の素材で、
モデルへ直接渡さない。出力は32実frameそれぞれのsource正規化座標 `(B,32,2)`。
公開境界は`model_io.mdd_pose.build_mdd_pose_detector`のmodel/adapter pairで、
`MDDPoseInput`を検証してから計算のみのforwardを実行する。
設定の正本は `../../configs/model/mdd_pose.yaml`。全重みをランダム初期化する。
refinerの候補/patch契約との接続は#986の未決事項で、座標からheatmapを捏造しない。

## MDD encoder

`encoder.py` は空間だけを各段1/2、合計1/8にする3段のencoder。
時間strideは常に1。各段の時間kernelは3で、最終MDD特徴の時間受容野は7frame
（中心±3frame）。窓端はfeatureのゼロpaddingで、RGB frameを反復しない。
正規化は各時空間位置のchannel LayerNorm。時間を跨ぐBatch/GroupNormは使わない。

| 方式 | 各段の処理 |
|---|---|
| `conv3d` | Conv3d kernel=3×3×3、stride=1×2×2、padding=1 |
| `average` | 2×2 spatial average → Conv3d kernel=3×1×1 |
| `unshuffle` | spatial PixelUnshuffle(2)、4C → Conv3d kernel=3×1×1 |
| `haar` | 1段の直交Haar DWT、LL/LH/HL/HH全保持、4C → Conv3d kernel=3×1×1 |

各段はchannel LayerNorm→GELUを続ける。最後に1×1×1投影でD次元へ写し、
各frameのH/8×W/8格子をpatch token列へflattenする。格子の正規化(x,y)を線形投影して加算する。
H,Wは8の倍数を要求し、勝手なresize/cropはしない。
Unshuffle/DWT自体は情報を保持するが、後続の学習可能なchannel投影まで可逆とはしない。

既定では `2→8→16→32→128` channels。720×1280画像なら
`360×640→180×320→90×160`、1frameあたり14,400 token。
32frameの最終patch tensorだけでfloat32約225MiBとなるため、batch/精度は学習前に決める。
MDD patch同士の大域self-attentionは行わない。

## pose・融合・座標

`pooling.py` は各frameの全人物を1 pose tokenへ集約する。
座標をsourceのW−1/H−1で正規化し、confidence/ID値は特徴として入力しない。
人物軸は可変で、clip内の固定player列を維持するが並べ替えにも不変。
欠測はmaskで除外し、人物0人/all-maskedには学習可能なnull tokenを使う。

- `deepsets`: 人物の17×2座標をMLP→人物間masked mean→MLP。
- `attention`: 人物MLP→1つのpose queryによるattention pooling。
- `hierarchical`: 関節埋込＋関節ID→人物内attention/pooling→人物間attention/pooling。
- `gnn`: COCO骨格edge＋self-loop、2段の正規化GCN→関節mean→人物attention pooling。

`model.py` の各blockは、pose（query条件ではposeとball queryをframeごとに交互配置）に
実PTS秒のRoPE付き時間self-attentionを行う。その後、同じframe内で
MDD patch→pose/queryへのcross-attention、pose/query→更新済みMDD patchへの
cross-attentionを順に行い、pose/query側へFFNを適用する。
時間attentionはoffline双方向。画像の局所時間受容野と、融合後のモデルの時間範囲は別。

`query`条件は各frameのball query、`pose`条件はpose tokenをreadoutにし、
LayerNorm→Linear(2)→sigmoidでuvを出す。pooling内部のpose queryはball queryとは別。
4圧縮×4pose集約×2readout＝32条件を同じ実装で選べる。

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
  --compression conv3d --pose-pooling attention --readout query
```

初期実装のlossはobserved uvのSmoothL1、val選択は重複frameを中心窓規則で
一度ずつ数えたsource画素の平均誤差。testはtrainerから読まない。
これらの設定も学習前レビューの対象であり、まだ実データ学習・性能評価は行っていない。

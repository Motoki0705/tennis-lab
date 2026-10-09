# Deep MDD CNNのヒートマップ事前学習

`config.py` / `encoder.py` / `model.py` は、CNNを共有するDPT事前学習モデルと
SwiGLU query-onlyモデルを提供する。学習は `scripts/pretrain_mdd_dpt.py` を入口にする。

## アーキテクチャ

RGB uint8 32枚 → 固定FP32 MDD → frameごとの残差CNN → 2D/2D/3Dの2段。
学習部にRGB・pose・courtの直接入力はない。学習可能な重みはランダム初期化。

| 空間stride | channel | 追加2D残差block |
|---|---:|---:|
| 2 | 16 | 0 |
| 4 | 32 | 1 |
| 8 | 64 | 2 |
| 16 | 128 | 2 |
| 32 | 192 | 1 |
| 64 | 256 | 1 |

各残差blockは3×3 Convを2層。encoderは2D Conv 22層、3D Conv 2層。
GroupNormはframeをbatchへ畳んで実施し、時間を跨いで統計を取らない。
空の先頭MDDでゼロ分散の正規化が連鎖しないよう、空間Convのbiasを学習し、
初期値もPyTorch標準のランダム初期化を使う。時間stride=1、3D kernel=3、
CNNのMDD時間受容野は前後2枚。空間の奇数サイズには通常のConv paddingを使う。

DPTは1/8・1/16・1/32・1/64の特徴を128chへ投影して粗い側から融合する。
`src/utils/models/dpt.py` のDPTを使用し、ballはbatch統計のない2層残差融合を選ぶ。
courtの従来wise-block設定・state_dict名は維持する。DPTはB×Tの各frameを独立に処理。
出力はsigmoid前logit `(B,32,ceil(H/4),ceil(W/4))`、720pでは180×320。
これはDPTの融合decoderをCNN特徴へ接続したモデルで、外部DPTの事前学習重みを使わない。
設計参考: [DPT](https://arxiv.org/abs/2103.13413)。

後段 `DeepMDDQueryDetector` は最終特徴へtoken投影・XY埋め込みを加え、
同時刻Cross → query列の時間RoPE Self → SwiGLUを4回通す。
dim256、8 heads、中間次元704（8/3 dimを共通ルールで64倍数に切り上げ）。
`transfer_encoder()` はschema・CNN全段の形・MDD契約を照合してCNNのみstrict転送し、
最初はCNNをfreezeする。後で `freeze_encoder(False)` により解凍できる。
後段decoder・位置埋め込み・head・optimizerをDPTから転送しない。

## データ・教師・評価

既存の固定MDD-only manifestを用い、プレイ候補内の32frame、元/半分/1/4 FPSを等分で混合。
RGBを間引いてからMDDを生成する。trainだけで重みを更新し、valで選択、testは使わない。
GTはsource端点で正規化した座標を出力heatmapの端点へ写すGaussian（sigmaは対角長×0.012）。
observedだけ正例、レビュー済みinstanceなし/out_of_frameをゼロheatmap負例とする。
未レビュー、補間、遮蔽推定、unresolved、reference、複数instanceはlossから除外。
複数instanceの除外は元の単一球coordinate manifest/readerに合わせた明示方針。

Focal BCE gamma2をFP32で計算し、画素平均→教師有効frame平均。推論はBF16、
decodeはsingle argmax＋log-parabolic subpixel補正。位置誤差はconfidenceで足切りせず、
全observed frameを数える。threshold0.5・8 source px一致のprecision/recall/F1も併記する。
既存ConvNeXtのmulti-peak/4px評価と同じ指標とは扱わない。

重複窓はFPSごとに中心に近い窓（同距離なら開始が早い窓）を所有者にする。
loss・位置誤差・検出指標とも所有者frameで集計。full/common subset・source/FPS別を出力。
bestは共通subsetのFPS等重み平均位置誤差で選び、単に空heatmapに近づくloss低下を選択理由にしない。

## 実行・保存

ローカルCUDA実行は必ずshared training queueを使う。CPU/FP32は小型テスト用。
CUDAはBF16固定。nvJPEG・画像先読み・preadv・upfront検証を既存入力経路から再利用する。

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.pretrain_mdd_dpt \
  --manifest /absolute/path/mdd_only_windows.json \
  --model-config /absolute/path/src/tasks/ball_detection/configs/model/mdd_dpt_pretrain.yaml \
  --output /absolute/path/new-output \
  --epochs 10 --windows-per-epoch 6000 --learning-rate .0002 --warmup-updates 500 \
  --device cuda --precision bf16 --batch-size 1 --seed 42 \
  --jpeg-decoder nvjpeg --image-prefetch --num-workers 8 --prefetch-factor 4 \
  --pin-memory --input-verification upfront --compile-mode default --preview-clips 3
```

AdamW、weight decay0.01、gradient clip1、warmup後はcosineでpeak LRの1/10へ。
epochごとにval・checkpoint・best.json・固定val clipのGIF（既定3clip）を保存。
GIFはGT緑・予測橙とheatmapを並べ、表示速度を100ms/frameに固定する。
`--resume` は同じ出力先・同じrecipe/実装の最新完了epochからのみ再開できる。
epoch途中の更新は再実行する。終了時に `COMPLETED.json` を保存し、
Transformerの学習は自動起動しない。

GPU smokeは `tests/benchmarks/ball_dpt_preflight.py`。本設定の実720p窓で
forward/backward、3D勾配、VRAM、validation、GIF、保存まで通す。
CNN転送・freeze/解凍、時間参照範囲、ignore/負例、重複排除はCPUテストでも確認する。

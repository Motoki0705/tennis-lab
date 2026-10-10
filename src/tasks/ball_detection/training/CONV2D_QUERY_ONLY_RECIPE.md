# Conv2d＋query-only：学習レシピ

2026-10-09更新。CUDA nvJPEG復号・モデル内FP32 MDD・BF16 compile・入力先読みに対応したレシピ。
開始・停止と診断結果は[研究の現在地](../../../../knowledge/summary.md#ball-detection)を参照。
BS・reader設定は実測で選び、学習率・総更新数は未学習モデルに対する初期レシピとして固定する。
GPU診断のloss低下は、本レシピの収束・精度を保証する根拠にしない。

## 入力・モデル

| 項目 | 固定値 |
|---|---|
| モデル | `conv2d-query_only`、1,126,762 parameters |
| 初期化 | 全重みをscratch。配布/既存checkpointからの転移なし |
| 外部入力 | RGB uint8、32枚、1280×720。モデル内で必ず固定FP32 MDDへ変換。pose/court入力なし |
| 学習部の画像入力 | MDD 2ch。RGBを直接使う学習経路なし |
| 空間stem | frame独立3×3 Conv2d、stride 2を4段。幅8→16→32→48、1/16 |
| 時空間block | 2D(stride 2)→2D→3Dを2回。幅64→96、最終1/64 |
| 時間方向 | Conv3dはkernel 3、stride 1。2層合成のMDD参照範囲はt−2〜t＋2 |
| MDD token | 12×20＝240個/frame、D=128、学習する2D位置embedding |
| query | 1個/frame。同じ初期query vectorを32時刻へ展開 |
| 融合 | 同時刻MDDへの片方向Cross→32 queryの時間Self-Attention→FFN、2 block |
| Attention | 4 heads、RoPE base=10,000、実PTS秒、dropout 0.1、FFN幅512 |
| 出力 | LayerNorm→Linear(2)→float32 sigmoid、frameごとの正規化uv |

時間attentionは32枚全体へ向ける。未来frameも読むoffline構成で、causalな遅延ゼロ推論ではない。
MDD tokenをqueryから更新する経路はない。1/64は特徴抽出gridで、座標出力の量子化単位ではない。

## データとサンプリング

入力の正本は`outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/`。
query-only用`mdd_only_windows.json`のSHA-256は
`035d3ab96807ace8e25ebe3a5d603f09a7c514802942170fdd5b8ad2854c949c`。
生成後に追加されたclip/poseをこのrunへ自動混入しない。

| split | 採用clip | 元FPS窓 | 1/2 FPS窓 | 1/4 FPS窓 |
|---|---:|---:|---:|---:|
| train | 824 | 20,661 | 10,047 | 4,640 |
| validation | 190 | 5,632 | 2,782 | 1,321 |
| test | 217 | 6,046 | 2,989 | 1,417 |

母集団1,300 clipのうち1,231 clipで窓を採用し、69 clipは窓条件を満たさない。
pose生成時のclip球存在率40%条件は使わない。pose推論・reviewや元splitは変更しない。

32枚の中の球存在証拠率50%以上、単一observed位置教師8枚以上を要求する。
native timelineのプレイ区間内に窓を収め、内部欠損は0.4秒以下だけを接続する。
参照専用frameと大きなPTS gapを跨がない。開始strideは間引き後16枚、区間末尾も含める。
窓不足を同一frameの反復で埋めない。

元FPS・1/2・1/4を同じrun内で等数混合し、各FPS内で窓をshuffleする。
source/clipの等数化はしないため、窓を多く持つsource/clipほど選ばれやすい。
不足するFPS群はshuffle cycleを繰り返す。epochごとのseedは`seed + epoch`。
ここでのepochは指定窓数の予算単位で、全窓を1回ずつ見ることを意味しない。

RGBを先にstep 1/2/4で選び、選択した32枚からMDDを再計算する。
必要なnative spanは32/63/125 frame。30FPS動画なら入力の先頭〜末尾は約1.03/2.07/4.13秒。
FPS名だけを変えたり、native MDDを間引いたりしない。
workerは選択したJPEGのbyte列だけを渡し、CUDA nvJPEGでRGB uint8へdecodeする。
次batchの復号を現在の学習と重ねる。モデル内で[0,1]への変換・輝度差・MDDをFP32計算し、
共通MDD実装の`a=0.2, b=0.15`を使う。
ImageNet正規化なし、窓先頭のMDDは0。PTS/教師/maskも同じ32枚へ揃える。
FPS混合と窓shuffle以外のaugmentation（flip/crop/色変換/時間反転等）は初回には入れない。

## 目的関数・最適化

| 項目 | 固定値 |
|---|---|
| 教師 | レビュー済み・対象frame・球が1個・point_kindがobservedのuvだけ |
| 正規化 | x/(source_width−1), y/(source_height−1)。保存画像のscaleを戻して対応 |
| 損失 | SmoothL1、beta=0.01。有効frameのx/y成分をbatch内で平均 |
| optimizer | AdamW、LR=1e−4、betas=(0.9,0.999)、eps=1e−8、weight decay=0.01 |
| decay対象 | norm/biasを含む全parameter。parameter group別の除外なし |
| LR schedule | 一定。warmup・schedulerなし |
| gradient clip | 全parameterのglobal L2 normを1.0以下へ。非finiteなら停止 |
| accumulation | なし。1 batch＝1 optimizer update |
| seed | 初回42。再現性確認用43/44の追加runは初回の結果を見て別途決める |

欠損・補間・遮蔽位置推定・未レビュー・複数球は座標loss/評価に使わない。
教師がないframeも時間文脈として入力に残るが、座標教師を捏造しない。
存在判定head/負例lossは持たず、どのframeにもuvを返す。非プレイ区間の偽陽性抑制性能を
この位置誤差だけで評価したとは扱わない。

## 評価と保存

ユーザー指定により、学習・評価のモデル演算はBF16 AMPに固定する。
重み・optimizer状態、MDD/PTS、座標出力・損失・座標誤差の計算はFP32を維持し、統計量の集計にはfloat64を使う。
各epoch後にvalidation全9,735窓をBF16で評価し、full/commonを両方保存する。
commonはposeあり側とGT・split・画像identityが一致した56 validation clip。
各FPS内の重複frameは「窓中央に近い予測、同点なら早い窓」を1つだけ採用する。
source画像pixel単位の平均・中央値・P95をFPS別、source別に記録する。

bestの選択はcommon validationの3つのFPS別平均誤差を等重み平均した値。
FPS内では教師frameの平均であり、sourceやclipを等重み平均する指標ではない。
fullの値とsource別P95も併記し、改善の偏りを確認する。testを選択・学習率調整に使わない。
レシピを固定してbestを選んだ後、test全10,452窓を1回評価し、fullとcommon（81 clip）を報告する。

各epochのmodel/optimizer/RNGを保存し、`best.json`で採用checkpointとhashを固定する。
train loss/LR/grad norm/速度は50 updateごと、validationと学習・評価時間はepochごとに記録する。
保存・再開の正確な契約は[共通の学習入口](COORDINATE_TRAINING.md#実行時設定中断からの再開)を参照。
早期停止による予算の自動変更は行わない。エラーや非finite値では停止し、原因を記録して判断する。

## GPU実行条件・予算

| 項目 | 採用案 |
|---|---|
| GPU | RTX 5060 Ti・16GBを共有queueの`resource=all`で使用 |
| precision | BF16 AMP、学習・validation・最終評価に同じ精度 |
| physical/effective BS | **1 / 1**（accumulationなし）。32枚は1窓の文脈で、独立な32 sampleではない |
| DataLoader | **8 workers、pin_memory=true、prefetch_factor=4、persistent_workers=true** |
| CPU | メインPyTorch 2 threads、各workerはPyTorch/OpenCV各1 thread |
| JPEG復号 | nvjpeg、RGB uint8、EXIF orientationなし。CUDA streamで1 batch先読み |
| 整合性 | upfront検証、成功をworker間共有。準備時間を別記し、学習中もstat変更を拒否 |
| Compile | Inductor `default`、fullgraph、静的shape、再compile上限8。backward autocast前提はoff |
| TF32/autotune | matmul TF32 off、cuDNN TF32 on、cuDNN benchmark off |
| 定期ログ | 50 updateごと。gradient normはclip前の値を記録 |
| ディスク | GPU診断でのepoch checkpointは約14MB。データsnapshotは既存のものを再利用 |

[1,020窓の最終確認](../../../../knowledge/nodes/ball_detection/000043-run-i986-pread-prefetch-long-20261009.md)に基づき、`preadv`による範囲読込と、8 workers×prefetch 4を使う。
同期／先読み・worker数・BS・メモリ内JPEG基準の比較は、[同一204窓の計測](../../../../knowledge/nodes/ball_detection/000036-run-i986-jpeg-prefetch-sweep-20261009.md)を参照。
[先読みの画素・順序検証](../../../../knowledge/nodes/ball_detection/000037-run-i986-jpeg-prefetch-integrity-20261009.md)と
[v4保存・再開・評価](../../../../knowledge/nodes/ball_detection/000051-run-i986-fenced-startup-cli-20261009.md)も確認した。

nvJPEGはOpenCVと異なる復号結果を持つ。ユーザー承認に基づく変更であり、
[RGB/MDD差分](../../../../knowledge/nodes/ball_detection/000033-run-i986-nvjpeg-pixels-20261009.md)を記録した。
画素一致や精度同等性を主張しない。train/val/testでは同じdecoder契約を使う。

学習予算は**60,000 optimizer updates＝6,000窓×10 epochs**、各FPS 20,000窓、
合計1,920,000入力frame（重複を含む）。毎epochに全9,735 validation窓を評価する。
LRと予算はユーザー承認済みの初期レシピで、速度測定から収束が保証された値ではない。
検証済み画像を再利用するbenchmarkの速度と、初回hash・compile・全validationを含む総時間を区別する。
本学習の長期安定性・汎化は実runで確認する。以前のGPU診断の90% allocator制限は、本学習prefix確認と本学習には適用しない。

## 実行コマンド

以下は本学習案のqueue登録例で、文書を作成しただけでは実行されない。
commitを固定した専用worktree rootから実行し、`--session`には実行するセッション自身のIDを指定する。
既存の固定bundleはそのまま使える。古い`experiments.json`にはcompile指定がないため、
ここに示すBF16/compile/decoder/先読み指定を使う。

```bash
cd /home/kamimura/projects/tennis-lab/.claude/worktrees/ball-nvjpeg-train-s42-u60000-v2
BALL_TRAIN_CMD=$(cat <<'COMMAND'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2 TORCH_LOGS=graph_breaks,recompiles .venv/bin/python -m src.tasks.ball_detection.scripts.train_mdd_pose \
  --manifest /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json \
  --model-config /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/models/conv2d-query_only.yaml \
  --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/train/conv2d-query-only-bf16/s42-u60000-nvjpeg-v2 \
  --device cuda --precision bf16 --compile-mode default --batch-size 1 \
  --num-workers 8 --pin-memory --prefetch-factor 4 --cpu-threads 2 \
  --jpeg-decoder nvjpeg --input-verification upfront --image-prefetch \
  --epochs 10 --windows-per-epoch 6000 --learning-rate 0.0001 --seed 42 \
  --mdd-a 0.2 --mdd-b 0.15 --selection-scope common --log-every 50
COMMAND
)
TRAINING_QUEUE_DIR=/home/kamimura/projects/tennis-lab/.training_queue \
  bash .agents/skills/training-queue/scripts/training_queue.sh add "$BALL_TRAIN_CMD" \
  --name i986-query-bf16-s42-u60000-nvjpeg-v2 --provider codex --session "$CODEX_THREAD_ID" \
  --issue 986 --resource all
TRAINING_QUEUE_DIR=/home/kamimura/projects/tennis-lab/.training_queue \
  bash .agents/skills/training-queue/scripts/training_queue.sh start
```

再開は上記の学習コマンドへ`--resume <同じoutput内の最新epoch-NNN.pt>`を加え、別queue job名で登録する。
同時に同じoutputへ書き込むjobを作らない。10 epochs終了後は`best.json`が指すcheckpointを固定して、
次のコマンドを同じ共有queueへ登録する。`<best-checkpoint.pt>`はその絶対パスに置き換える。

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.evaluate_mdd_coordinates \
  --checkpoint <best-checkpoint.pt> \
  --manifest /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json \
  --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/evaluate/conv2d-query-only-bf16/s42-u60000-nvjpeg-v2/test.json \
  --split test --device cuda --precision bf16 --compile-mode default --batch-size 1 \
  --num-workers 8 --pin-memory --prefetch-factor 4 --cpu-threads 2 \
  --jpeg-decoder nvjpeg --input-verification upfront --image-prefetch
```

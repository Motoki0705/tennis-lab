# Conv2d＋query-only：初回学習レシピ

2026-10-08時点の提案。GPU診断と本学習を区別する。
BS・reader設定は実測で選び、学習率・総更新数は未学習モデルに対する初期レシピとして固定する。
GPU診断のloss低下は、本レシピの収束・精度を保証する根拠にしない。

## 入力・モデル

| 項目 | 固定値 |
|---|---|
| モデル | `conv2d-query_only`、1,126,762 parameters |
| 初期化 | 全重みをscratch。配布/既存checkpointからの転移なし |
| 入力 | MDD 2ch、32枚、保存解像度1280×720。pose/RGB画像/courtの直接入力なし |
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
JPEGのRGBを[0,1]へ変換して輝度差を取り、共通MDD実装の`a=0.2, b=0.15`を使う。
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
| DataLoader | **8 workers、pin_memory=true、prefetch_factor=1、persistent_workers=true** |
| CPU | メインPyTorch 2 threads、各workerはPyTorch/OpenCV各1 thread |
| TF32/autotune | matmul TF32 off、cuDNN TF32 on、cuDNN benchmark off。torch.compileなし |
| 定期ログ | 50 updateごと。gradient normはclip前の値を記録 |
| ディスク | GPU診断でのepoch checkpointは約14MB。データsnapshotは既存のものを再利用 |

同じ108窓で比較した結果は以下。速度は先頭12窓を除いた96窓のwall-clock平均。

| BF16 BS | workers | 窓/秒 | 最大reserved VRAM |
|---:|---:|---:|---:|
| 1 | 4 | 1.22 | 2.40GiB |
| 1 | 6 | 1.58 | 2.40GiB |
| **1** | **8** | **1.85** | **2.40GiB** |
| 2 | 4 | 1.34 | 4.76GiB |
| 2 | 6 | 0.86 | 4.76GiB |
| 2 | 8 | CUDA unknown errorで失敗 | 未確定 |
| 4 | 4 | 0.71 | 9.48GiB |

BS=1・worker=8は、別の252窓連続実行でも成功し、warmup後240窓では**1.33窓/秒**だった。
異なる窓列・clip初回検証・OS cacheによって速度は変わる。GPU常駐だけなら約7窓/秒だが、
本学習の時間見積りには使わない。VRAMはPyTorch reservedであり、表示/CUDA context分は別。
GPU診断はallocatorを90%に制限して実施した。本学習CLIはその上限を変更せず、
推奨BSが実測で十分に小さいことにより余裕を保つ。

BF16 BS=8はGPU常駐計測でOOM。BS2/worker8の別のCUDA unknown errorは原因未特定で、
OOMとは断定しない。この組合せは採用しない。推奨設定では通常CLIの学習→保存→再開→
BF16評価（実train/val各3 clip、計24 update）も通過した。
数日規模の本学習の安定性・収束はまだ検証していない。

最初の精度確認と本学習の提案予算を分ける。いずれも**今回のGPU診断では起動していない**。

| 段階 | 更新数 | epochの定義 | validation | 目的 |
|---|---:|---|---|---|
| pilot | 3,000 | 3,000窓×1 epoch | 全9,735窓を最後に1回 | LR=1e−4で学習が進むか、full/common/source別の誤差・外れ値を確認 |
| 本学習案 | 60,000 | 6,000窓×10 epochs | 毎epoch、全9,735窓 | 初回の収束曲線とbestを取得 |

本学習案は各FPS 20,000窓、合計1,920,000入力frame（重複を含む）。
pilot後はレシピを評価し、本学習は別runとしてscratchから開始する。
pilotと本学習で窓/epochが違うため、pilot checkpointを本学習へresumeする設定ではない。
LRと60,000 updateは初期提案であり、速度測定から最適値が判明したわけではない。
pilotで学習が停滞・発散する場合は原因を検討して新しいレシピを作り、未承認の長期runへ自動移行しない。

実測1.33〜1.85窓/秒を使うと、train部分はpilot約27〜38分、本学習約9.0〜12.5時間。
validationの通し速度は未測定。仮に同じ窓/秒を使うとvalidation 1回は約88〜122分で、
本学習案のtrain＋10回validationは約24〜33時間となる。validationは逆伝播がなくclip順に読むため
実際には異なり、これを完了時刻の保証にはしない。起動時検証・保存・queue待ちは別。

計測の正本は[同一窓比較](../../../../knowledge/nodes/ball_detection/000025-run-i986-query-bf16-confirm-20261008.md)、
[実CLI検証](../../../../knowledge/nodes/ball_detection/000026-run-i986-query-bf16-cli-20261008.md)、
[連続実行](../../../../knowledge/nodes/ball_detection/000027-run-i986-query-bf16-sustained-20261008.md)。

## 実行コマンド

以下は本学習案のqueue登録例で、文書を作成しただけでは実行されない。
実装worktree rootから実行し、`--session`には実行するセッション自身のIDを指定する。
pilotは出力先を別名にし、`--epochs 1 --windows-per-epoch 3000`へ変更する。
生成時の古い`experiments.json`テンプレートには精度指定がないため、ここに示すBF16指定を使う。

```bash
cd /home/kamimura/projects/tennis-lab/.claude/worktrees/ball-query-only-training
BALL_TRAIN_CMD=$(cat <<'COMMAND'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m src.tasks.ball_detection.scripts.train_mdd_pose \
  --manifest /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json \
  --model-config /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/models/conv2d-query_only.yaml \
  --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/train/conv2d-query-only-bf16/s42-u60000-v1 \
  --device cuda --precision bf16 --batch-size 1 \
  --num-workers 8 --pin-memory --prefetch-factor 1 --cpu-threads 2 \
  --epochs 10 --windows-per-epoch 6000 --learning-rate 0.0001 --seed 42 \
  --mdd-a 0.2 --mdd-b 0.15 --selection-scope common --log-every 50
COMMAND
)
TRAINING_QUEUE_DIR=/home/kamimura/projects/tennis-lab/.training_queue \
  bash .agents/skills/training-queue/scripts/training_queue.sh add "$BALL_TRAIN_CMD" \
  --name i986-query-bf16-s42-u60000-v1 --provider codex --session <current-session-id> \
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
  --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/evaluate/conv2d-query-only-bf16/s42-u60000-v1/test.json \
  --split test --device cuda --precision bf16 --batch-size 1 \
  --num-workers 8 --pin-memory --prefetch-factor 1 --cpu-threads 2
```

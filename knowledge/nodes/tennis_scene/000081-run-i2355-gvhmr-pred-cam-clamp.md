---
id: run-i2355-gvhmr-pred-cam-clamp
type: run
task: tennis_scene
sequence: 81
recorded_at: '2026-11-22'
title: GVHMR TensorRT FP16 B>1ビルド異常終了の切り分けと加算mask回避
issue: 2355
provider: claude
date: '2026-11-22'
status: done
summary_impact: none
config:
  gpu: RTX 5060 Ti (CC 12.0)
  tensorrt: 10.14.1.48.post1
  network: strongly typed, TF32 off, workspace 512 MiB
  diagnosis_shape: {batch: 2, length: 145}
  measured_batches: [1, 2, 4]
  measured_lengths: [145, 177]
  trt_min_opt_max_batch: [1, 4, 4]
  trt_min_opt_max_length: [145, 145, 177]
  attention_mask_rewrite: additive-finite (scores + Where(mask, -32768, 0))
  accuracy_reference: PyTorch FP32 batch 1 (#2335)
metrics:
  fp16_b2_block0_failing_cases: 13
  fp32_b2_block0_failing_cases: 0
  fp16_b1_block0_failing_cases: 0
  builder_option_probes_failing: 10
  builder_option_probes: 10
  additive_inf_full_graph_build_exit: 0
  additive_inf_l177_nonfinite_pred_x_b1: 26727
  additive_finite_fp32_bit_equal_to_source_onnx: true
  additive_finite_fp16_bit_equal_shapes: 6
  additive_finite_fp16_layers_verified: 110
  boundary_probes_verified: 6
  gvhmr_trt_fp16_additive:
    l145_b1_sequences_per_second: 515.1
    l145_b2_sequences_per_second: 995.2
    l145_b4_sequences_per_second: 1783.6
    l177_b1_sequences_per_second: 459.4
    l177_b2_sequences_per_second: 923.7
    l177_b4_sequences_per_second: 1607.8
    l145_incam_joint_p95_mm: 8.78
    l177_incam_joint_p95_mm: 8.99
    l145_global_joint_p95_mm: 24.46
    l177_global_joint_p95_mm: 24.09
artifacts:
  run_dir: knowledge/runs/run-i2355-gvhmr-pred-cam-clamp
  diagnosis: knowledge/runs/run-i2355-gvhmr-pred-cam-clamp/diagnosis-summary.json
  ort_variant_accuracy: knowledge/runs/run-i2355-gvhmr-pred-cam-clamp/ort-variant-accuracy.json
  campaign: knowledge/runs/run-i2355-gvhmr-pred-cam-clamp/records/campaign-additive-finite-1.json
  queue_outputs: /mnt/wsl/tennis-data/home/kamimura/projects/tennis-lab/.worktree-outputs/parent-2278-issue-2355-sol56b/trt-gvhmr-i2355
parents:
- run-i2335-trt-batch-sweep
- run-i2305-gvhmr-tensorrt
relations: []
papers: []
tags: []
---

## 考察 / Findings

#2335では、RTX 5060 TiとTensorRT 10.14.1でGVHMR FP16のB>1ビルドが異常終了した（終了値 -11）。これは `pred_cam` の `ScatterND` ではなく、最初のTransformer blockのattentionで起きていた。maskの書き方を変えた等価グラフでは、B=1〜4・L=145〜177の動的planを作れた。全演算がFP16のまま、B=4で #2305 のB=1専用planの約4.5倍の系列/sが出た。精度はB=1専用planやPyTorch FP16と同程度だった。手順とファイルは [bundle README](../../runs/run-i2355-gvhmr-pred-cam-clamp/README.md) にある。

### 切り分け

各ビルドは別の子プロセスで行い、終了値だけを記録した。静的shapeはL=145、builder最適化レベルは3（注記したものを除く）。

| 比較 | 結果 |
|---|---|
| `pred_cam` の `ScatterND` を含む46ノードの最小グラフ（FP16/FP32、B=1/2） | すべて終了値0 |
| `ScatterND` を `Slice`/`Clip`/`Concat` に書き換えた実モデル（FP32 B=2でPyTorchと一致） | 動的FP16 planは -11 |
| 実モデルFP16 B=2を、mask生成・embedding・`pred_cam` headだけに切る | 0 |
| 同じく、最初のblockまでのprefix | -11 |

最初のblockを演算単位で切った27ケースの結果は次のとおり。 [ケースごとの終了値と最後のForeignNode](../../runs/run-i2355-gvhmr-pred-cam-clamp/diagnosis-summary.json)

| 出力の境界（block入力と2つのmaskを入力にした場合） | FP16 B=2 | FP32 B=2 | FP16 B=1 |
|---|---:|---:|---:|
| LayerNorm、Q/K + RoPE、QK^T/8、局所mask、padding mask、Softmax | 0 | 0 | 0 |
| P·V（`attn/MatMul_1`）以降、block出力まで | -11 | 0 | 0 |
| QK、mask、Softmax、P·Vの各演算だけ | 0 | 0 | 0（3件は未ビルド） |
| 最初のblockまでのprefix、最適化レベル0 | -11 | 0 | 0 |

FP16 B=1の演算単体3ケースは終了値1で、ビルドしていない。昇格した入力のbatch次元が記号のまま残り、静的shape検査が意図どおり止めた。

FP16 B=2のprefixで、builder設定10種もすべて -11だった。最適化レベル1/2/4/5、tactic source無効、aux stream 0、tactic 1個、tiling FAST、`DISTRIBUTIVE_INDEPENDENCE`、compilation cache無効を試した。設定の調整では回避できない。

最小の失敗範囲は、Q/K/Vと2つのmaskを入力にしてP·Vまでを計算する部分グラフだった（QK^T、/8、局所mask、padding mask、Softmax、P·V）。元のexportでは32ノード、`Expand` を除くと14ノードになる。FP16でB=2のときだけ異常終了した。QK^T/8の出力から始めると0、mask適用後のscoreから始めても0だった。最後のTiming Runnerは `ForeignNode[/network/blocks.0/attn/Reshape_8.../network/blocks.0/attn/MatMul_1]` で、入力はHalf、Bool、Boolだった。

### 回避策

静的FP16 B=2の同じ境界でビルドを比べた。

| 変種 | 等価性 | P·V（block入力から） | block出力 | prefix | 全graph |
|---|---|---:|---:|---:|---:|
| 元（`Where(Cast(Expand(mask)), -inf, scores)`） | — | -11 | -11 | -11 | -11 |
| `Expand` と `Cast` の除去 | ORTでビット一致 | -11 | -11 | -11 | -11 |
| 加算mask `scores + Where(mask, -inf, 0)` | ORTでビット一致 | 0 | 0 | 0 | 0 |
| 局所maskだけ除去（診断用、非等価） | — | 0 | 0 | 0 | 0 |
| padding maskだけ除去（診断用、非等価） | — | 0 | 0 | 0 | 0 |
| 両mask除去（診断用、非等価） | — | 0 | 0 | 0 | 0 |
| P·VだけFP32 | 丸めが変わる | 0 | -11 | -11 | -11 |
| SoftmaxだけFP32 | 丸めが変わる | 0 | 0 | 0 | 0 |
| Softmax + P·VをFP32 | 丸めが変わる | 0 | 0 | 0 | 0 |
| attention全体（QK〜P·V）をFP32 | 丸めが変わる | 0 | 0 | 0 | 0 |

観測した範囲では、2つのBool maskが同じattentionにそろい、SoftmaxがFP16のときだけ異常終了した。SoftmaxをFP32にした変種でも、TensorRTは12個のfused MHA kernelを使い、Bool maskを受け取っていた。fused MHAを使うこと自体は条件ではない。

FP32固定の変種は、ONNX RuntimeのFP16 B=2でFP32グラフとの差が元のFP16グラフ以下だった（`pred_x` 最大0.0306 / 0.0316）。 [変種ごとの差](../../runs/run-i2355-gvhmr-pred-cam-clamp/ort-variant-accuracy.json)

### 加算maskの動的plan

最初の `-inf` 加算maskでは、B=1〜4・L=145〜177の動的planを作れた。しかし境界probeで、L=177の出力が非有限になった。B=1でも系列全体の `pred_x` がNaNで、ONNX Runtimeは有限だった。L=145はB=1/2/4で有限で、ORTとの差は `pred_x` 最大0.026。 [-inf版のshape別出力](../../runs/run-i2355-gvhmr-pred-cam-clamp/records/inspect-additive-inf.json)

このplanは測らなかった。1回目の試行では層精度検査が止めた。動的profileのengine inspectorが次元を -1 で返し、K=2の `learned_pos_linear` はFP16のスカラー積和に下げられていたためである。検査器をこの2つに対応させた2回目で、境界probeがL=177の非有限値を検出した。 [1回目](../../runs/run-i2355-gvhmr-pred-cam-clamp/records/campaign-additive-inf-attempt1.json)・[2回目](../../runs/run-i2355-gvhmr-pred-cam-clamp/records/campaign-additive-inf-attempt2.json)

maskの値を `-inf` から `-32768` に変えた（`additive-finite`）。scoreが有限の範囲では、ORTの結果はビット一致した。確認したのは次のとおり。

- FP32静的B=2・L=145で、全5出力が書換え前のONNXとビット一致した。PyTorch FP32との差は `pred_x` 5.0e-6。
- FP16動的グラフで、B=1/2/4 × L=145/177の6 shapeが書換え前とビット一致した。
- planのmin/opt/maxで、110個の主要演算がFP16で計算された。そのうち12個は加算mask付きのfused MHA、1個はK=2のFP16積和。
- B=1/L=145とB=4/L=177は出力が有効で、B=0/5、L=144/178はTensorRTが拒否した。

### 計測

同じ固定入力を、#2305と同じ精度照合、#2335と同じ計測harnessで測った。関節差はPyTorch FP32 B=1に対するCOCO-17のP95。条件2のB=1は #2305 のB=1専用planで、#2335の表から転記した。PyTorch FP32は #2335 の値。 [全計測と証跡](../../runs/run-i2355-gvhmr-pred-cam-clamp/records/campaign-additive-finite-1.json)

| L | plan | B | 中央値 / P95 (ms) | 系列/s | frames/s | allocator / NVML増分 (MiB) | nsys kernel | カメラ / 重力整列関節差 (mm) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 145 | PyTorch FP32（#2335） | 4 | 15.25 / 21.82 | 242.9 | 35,223 | 220.3 / 407.1 | 713 | 0.004 / 1.397 |
| 145 | TRT FP16 B=1専用（#2305） | 1 | 2.51 / 3.17 | 393.6 | 57,068 | 0.3 / 263.1 | 135 | 8.614 / 23.499 |
| 145 | TRT FP16 加算mask | 1 | 1.81 / 2.44 | 515.1 | 74,697 | 0.3 / 267.1 | 135 | 8.777 / 24.455 |
| 145 | TRT FP16 加算mask | 2 | 1.92 / 2.53 | 995.2 | 144,303 | 0.7 / 267.1 | 135 | 8.777 / 24.455 |
| 145 | TRT FP16 加算mask | 4 | 2.19 / 2.56 | 1,783.6 | 258,616 | 1.4 / 287.1 | 135 | 8.777 / 24.455 |
| 177 | PyTorch FP32（#2335） | 4 | 16.37 / 23.58 | 232.8 | 41,213 | 224.4 / 407.1 | 745 | 0.016 / 1.594 |
| 177 | TRT FP16 B=1専用（#2305） | 1 | 2.50 / 2.98 | 394.2 | 69,780 | 0.4 / 263.1 | 135 | 8.721 / 18.325 |
| 177 | TRT FP16 加算mask | 1 | 1.95 / 3.18 | 459.4 | 81,311 | 0.4 / 267.1 | 135 | 8.991 / 24.088 |
| 177 | TRT FP16 加算mask | 2 | 1.87 / 4.76 | 923.7 | 163,497 | 0.8 / 267.1 | 135 | 8.991 / 24.088 |
| 177 | TRT FP16 加算mask | 4 | 2.40 / 2.77 | 1,607.8 | 284,578 | 1.7 / 287.1 | 135 | 8.991 / 24.088 |

全6ケースで、長さとmaskはFP32と一致し、非有限値はなかった。復号後出力のSHA-256は、各LでB=1/2/4が同一だった。同じplanでは、束ねる人数で値が変わらない。

カメラ座標の関節差は8.78 / 8.99 mmで、B=1専用plan（8.61 / 8.72）やPyTorch FP16（8.45〜8.65）と同程度だった。重力整列座標の24.46 / 24.09 mmは、PyTorch FP16の範囲（16.47〜24.05）の上端付近にある。L=177のB=1専用plan（18.33）より大きい。planとtacticの違いによるFP16の丸め差と考えられるが、検証していない。

### Phase 2（#2362）への含意

- GVHMR TRTのB=1制限は、次の条件を満たせば外せる。`masked_fill(-inf)` を、有限値の加算mask（`scores + Where(mask, -32768, 0)`）としてexportまたは書き換えること。書換え前とのビット一致をONNX Runtimeで確かめること。
- `-inf` の加算maskはビルドできたが、L=177で非有限値を出した。そのため不可とする。境界probeの有限性検査を必ず残す。
- 有限値で埋めた書換えは、softmaxの行に未maskの要素が残る限りビット一致する。全要素がmaskされた行では、元はNaN、書換えは一様分布になる。今回の入力にはそのような行はなかった。系列長に満たないpaddingを含むバッチは測っていない。
- 他の回避策として、SoftmaxをFP32に固定すれば静的B=2の全graphをビルドできた。動的planの計測はしていない。
- 層精度検査は `additive_precision.py` の拡張を前提とする。共有の `verify_compute_precision` は、加算mask付きMHAとK=2のFP16積和を拒否する。

### 未確認

- 異常終了の内部原因。ログ、終了値、部分グラフの範囲だけを記録し、プロセスのメモリやスタックは調べていない。NVIDIAの公開Issueとの照合は `ScatterND` の条件で行ったもので、今回の範囲（Bool mask 2つ + FP16 Softmaxのfused attention、B≥2）では照合していない。
- `-inf` 加算maskがL=177だけで非有限値を出す理由。
- 長さの異なる系列を同じバッチにpaddingして詰めた場合の精度。
- ncuの内部指標、標準パイプラインでのE2E速度と下流精度。

### summaryとの関係

summary.mdの判断は変えない。TensorRTの記述は人物モデル単体比較に限られ、GVHMRのバッチ推論の採否はまだ書かれていない。今回の結果は、#2362のE2E評価で採否を決める根拠として、このnodeから参照する。

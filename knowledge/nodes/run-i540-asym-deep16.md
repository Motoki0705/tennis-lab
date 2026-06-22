---
id: run-i540-asym-deep16
type: run
title: i540_asym_deep16
issue: 535
provider: claude
session: d22b7d68-7d91-4a6f-862d-434085e5d2d9
date: '2026-06-20'
status: done
config:
  model: multiview_axial_split_asym_deep16
  loss: canonical_rot
  data: multiview_sequence
metrics:
  position_error_m: 0.207075
  position_error_std_m: 0.182874
  position_error_median_m: 0.153868
  angular_error_deg: 8.396061
  angular_error_std_deg: 7.641904
  angular_error_median_deg: 6.27077
  x_error_m: 0.076728
  y_error_m: 0.171573
  z_error_m: 0.034041
  position_accuracy: 0.950193
  angle_accuracy: 0.839364
  position_accuracy_0.5m: 0.950193
  position_accuracy_1m: 0.991129
  position_accuracy_2m: 1.0
  angle_accuracy_10deg: 0.689944
  angle_accuracy_15deg: 0.839364
  angle_accuracy_30deg: 0.984475
repro:
  commit: 6399aa6f6848994957412eebabf4a4330c95cd15
  branch: feat/issue-533-experiment-log-format
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.plcs.scripts.train
    model=multiview_axial_split_asym_deep16 loss=canonical_rot data=multiview_sequence
    training.trainer.max_epochs=200 run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i540-asym-deep16
  predictions: knowledge/runs/run-i540-asym-deep16/pred_test.npz
  log: .training_queue/logs/1781927908633902412_762455_i540_asym_deep16.log
  curves: knowledge/runs/run-i540-asym-deep16/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_split/logs/version_5
parents:
- run-i525-asym
- run-i518-exp10
relations:
- to: run-i535-asym-deep16-rerun
  rel: supersededby
tags:
- plcs
- canonical
- split-trunk
- asymmetric
- depth
- capacity-frontier
- retracted-noop
---

## 考察 / Findings

> ⚠️ **【無効・撤回 2026-06-21】このランは作業ディレクトリ取り違えによる no-op 計測です。**
> `cd /home/kamimura/projects/tennis-lab`（main ツリー）で実行されたため `rot_num_task_layers` が黙って無視され、
> 意図した rot16/512（142.3M）ではなく**対称 6 層/512 = EX10 の再学習（78.1M）**になっていました（params が EX10 と一致したのはこのため）。
> 8.40° は EX10 のばらつき範囲内の値で「深さが効いた」証拠ではありません。正しい worktree（`exp/i525-asym`）での有効な再計測は
> **[[run-i535-asym-deep16-rerun]]**（142.3M / 10.41° / 0.252m ＝ EX10 に届かず）で、本ノードを supersede します。
> **以下の各節は撤回済みの当初考察**で、記録のために残しています。

### 要約
rotation trunk を 16 層まで深めた非対称構成（200ep）。位置 0.207m / 回転 8.40° で本実験群の絶対最良、同等予算で EX10 を初めて両指標とも上回る。（※上記のとおり no-op で実際は対称 EX10 を再学習しており本結論は無効）

### アーキテクチャ詳細
`multiview_axial_split_asym_deep16` + `canonical_rot`：pose trunk 6 層・**rotation trunk 16 層**、`hidden_dim 512` / `num_heads 8`、約 78.1M params（EX10 と同等予算）。`max_epochs=200`。

### メトリクスの解釈
位置 `0.207m` / 回転 `8.40°`。回転は EX10 (`9.98°`)、位置も EX10 (`0.238m`) を上回る。中央値も回転 `6.27°`・位置 `0.154m`、`angular_std 7.64°` で外れ値依存ではない。

### アーキテクチャ⇄メトリクスの因果考察
「分離 rotation trunk の深化は回転に効く」が成立。[[run-i525-asym]]（rot=10, 103M）が `19.94°` と劣化した主因は深さそのものではなく、103M という過大容量が 200ep で未収束 / 最適化困難だった可能性が高い。深さは「学習可能なサイズ envelope に収まる限り」回転の主レバー。さらに幅広の [[run-i540-asym-wide]]（768 幅, 172M, `12.27°`）より安価かつ高精度で、幅 < 深さ。

### 既存実験との比較
[[run-i525-asym]] の負の結論を覆す（`contradicts`）。[[run-i518-exp10]] を同予算で超える（`compares`）。深さ振りの [[run-i541-parameff-deeppose]] と同結論（`confirms`）。

### 次に有効な実験
deep16 が新ベースライン候補。(1) rotation 深さの最適点（12/16/20 層）を sweep、(2) hidden_dim を 512→384 に絞っても深さで回転を維持できるか（deeppose 系と接続）を確認したい。

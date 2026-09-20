---
task: slcs
sequence: 28
recorded_at: 2026-09-18
date_source: experiment_date
papers: []
id: run-slcs-meiji-root-support-boundaries-v1
type: run
title: 'Meiji肩支持の副作用: view切替境界と速度棄却'
provider: codex
date: '2026-09-18'
status: done
config:
  diagnosis: 'CPU: current hips vs hips_and_shoulders; final speed rejection transitions'
  data: three affected Meiji v7 player1 clips
  max_speed_mps: 12.0
  min_confidence: 0.3
metrics:
  affected_player_clip_pairs: 3
  new_speed_rejected_frames: 35
  resolved_speed_rejected_frames: 17
  net_additional_speed_rejected_frames: 18
  control_max_position_diff_from_saved_m: 9.5367431640625e-06
  new_rejected_frame_classes:
    B-boundary-switch: 32
    S-2view-jitter: 3
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-root-support-boundaries-v1
  output_dir: outputs/tennis_scene/analyze/meiji_root_support_probe/s42-002
  diagnostics: knowledge/runs/run-slcs-meiji-root-support-boundaries-v1/speed_reject_audit.json
parents:
- run-slcs-meiji-root-support-probe-v1
relations: []
tags:
- slcs
- meiji
- quality-control
- cpu-diagnosis
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
repro:
  commit: 777d339101e162739c12e37cf52dbb4fe75cce6a
  branch: codex/slcs-real-rgb
  command: .venv/bin/python knowledge/runs/run-slcs-meiji-root-support-boundaries-v1/01_speed_reject_audit_cpu.py
  patch: knowledge/runs/run-slcs-meiji-root-support-boundaries-v1/uncommitted.patch
---

## 考察 / Findings

### 要約
肩支持候補で増えた速度棄却の局所要因を分解した。新規棄却35frameのうち32frameは参加viewが3↔2へ切り替わる境界、3frameは三角測量内部の速度棄却と最終gateの履歴差だった。解消17frameと相殺し、正味18frame（対象3clipで各6frame）増える。

### アーキテクチャ詳細
前runの既存raw観測・camera fitを使い、同じhip中心・confidence0.3・速度12m/s・補間設定でrootをCPU再計算した。記録済みrefinedとcontrolのsourceは一致、最大位置差は約9.54e-6mの丸め範囲。現行データ・weightは変更しない。実行時未追跡だったscriptをそのまま同梱し、記載commitへ適用する追加patchを保存した。

### メトリクスの解釈
001/001 P1は新規18/解消12、000/007 P1は9/3、000/009 P1は8/2。001/001の168→169frameは現行0.52m/s、候補69.57m/sで、candidateだけ3→2viewへ切り替わる。338→339の2view内部は候補0.32m/s。frame333のcam0/1残差は現行7.72/3.17px、候補0.71/0.44pxだった。学習を伴わず曲線なし。

### アーキテクチャ⇄メトリクスの因果考察
今回の追加棄却は主に異なる観測集合の解が切り替わる段差で説明できる。残存2viewの観測整合性が良好な例はあるが、その同じ観測から作った位置なので、正しい実測3D・校正誤差なし・普遍的な2view安定性までは証明しない。速度gateを緩めず境界両端のweightを0にすることは、不連続な教師を学習へ渡さないための意図した挙動と判断する。

### 既存実験との比較
親runでは正味棄却増だけが分かっていた。本runで増加・解消と発生境界を分離した。肩条件は部分切れcam2による約1mの引込みを減らす一方でview切替境界を追加する。全体への外挿はまだ行わない。

### 次に有効な実験
肩支持を新しいMeiji v8に明示し、既存4clipで本実装を候補と照合する。固定速度・再投影・coverage閾値で全clipを評価し、境界は除外を保つ。平滑化やヒステリシスによる未検証の補完は追加しない。

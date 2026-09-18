---
id: run-slcs-meiji-root-support-probe-v1
type: run
title: 'Meiji腰rootの観測支持: 肩を要求するCPU比較'
provider: codex
date: '2026-09-18'
status: done
config:
  diagnosis: CPU replay of existing raw scene; root triangulation only
  control: both hips confidence >=0.3
  candidate: both hips and both shoulders confidence >=0.3
  shared: unchanged reprojection/speed/bounds/interpolation thresholds
  dataset: meiji_rgb_v7 four completed clips
metrics:
  completed_clips: 4
  player_clip_pairs: 8
  control_max_position_diff_from_saved_m: 9.5367431640625e-06
  candidate_displacement_over_1m_p1_frames: 44
  candidate_displacement_over_1m_p0_frames: 0
  candidate_max_displacement_m_rounded: 1.31
  three_views_to_below_two_count: 0
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-root-support-probe-v1
  output_dir: outputs/tennis_scene/analyze/meiji_root_support_probe/s42-001
  diagnostics: knowledge/runs/run-slcs-meiji-root-support-probe-v1/root_support_probe.json
parents:
- run-slcs-meiji-temporal-probe-v1
relations: []
tags:
- slcs
- meiji
- quality-control
- cpu-diagnosis
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
repro:
  commit: 7a452096bddef69a27a7fdcb9281c9c500e5531c
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: .venv/bin/python knowledge/runs/run-slcs-meiji-root-support-probe-v1/01_root_support_probe_cpu.py
  patch: knowledge/runs/run-slcs-meiji-root-support-probe-v1/uncommitted.patch
---

## 考察 / Findings

### 要約
部分切れのcam2でhip scoreが0.3を跨ぐと、ほぼ不変のcam0/1観測に対して腰rootが約1m変動する機構を確認した。両肩もconfidence0.3以上とする候補はその例を改善するが、他の区間の速度棄却は増えるため、まだ本番教師へ採用していない。

### アーキテクチャ詳細
既存raw sceneからroot三角測量・短区間補間・12m/sの両端重み0処理をCPUで再計算した。現行レプリカは4clip×2選手で保存済みrefinedと最大位置差9.53674316e-06m（浮動小数点丸め）、source/weight完全一致。候補は三角測量に使うviewへ左右肩(5,6)のscore条件を追加するだけで、hip中心・幾何閾値・補間・速度閾値は同じ。ball/yaw/正準姿勢・既存datasetは変更しない。

### メトリクスの解釈
001/001 P1の291/292・294〜297frameはcam2両hip score0.30〜0.38、両肩0、hip x約0.8〜2.5pxだった。現行の3view腰Y約14.96〜14.98mに対し、候補の2view腰Yは約15.95〜15.96m。cam0/1のhip残差は約8.6/3.1pxから約1.0/0.6pxへ改善する。変位1m超はP1で44frame、P0は0frame。3viewから2view未満になったframeは0。

教師frame率は001/001 P1で約0.9815→0.9769、000/009 P1で約0.9866→0.9772。最終速度棄却は001/001 P1で24→30、000/007で7→13、000/009で10→16と増加した。3→2viewは001/001 P1で68frame、他P1で7〜15frame。利用率と観測整合性の比較で、実測3D精度ではない。学習を伴わず収束曲線なし。

### アーキテクチャ⇄メトリクスの因果考察
同じ保存観測・同じ算法でview支持条件だけを変えているため、この例の位置変動が部分切れcam2の出入りに依存することは示せる。肩支持の追加はこのケースで有効だが、2本の視線になる幾何的な情報減少、他区間の速度棄却増、近似校正への依存が残る。候補を汎用的な精度改善と断定しない。

### 既存実験との比較
時間的な人物対応は通行人への切替を改善したが、部分切れ関節のconfidenceに起因するviewの不安定さは別の課題だった。画像内外の診断で見つけた腰の大残差frame300/303は短区間補間でcam0/1に整合しており、全関節maxだけで教師を除外する方法とは区別する。

### 次に有効な実験
新しいデータ版で肩支持条件を明示し、既存4clipで速度棄却増の局所像を確認してから全clipへ適用を判断する。現行datasetと規約・重みを混在させない。checksum問題の切り分けは別実験として継続する。

実行時の診断scriptは未追跡だったため、同梱scriptとその追加patchを保存した。再現時は記載commitへpatchを適用してからcommandを実行する。絶対入力・出力pathを含む当時の環境を前提とする。

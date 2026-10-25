---
id: run-court-scene-diversity-6000-v1
type: run
task: synthetic_data_generation
sequence: 28
recorded_at: '2026-10-25'
title: 既存train軌道を固定した6000枚scene数比較データのローカル3DGS生成
provider: codex
session: 01a171f1-ae7a-7d53-b09a-135549f91d0b
date: '2026-10-25'
status: done
config:
  method: saved_train_trajectories
  generation_seed: 42
  source_sampling_seed: 695
  source_experiment: group-i983-court-dinov3-shared1024-s42
  density_level:
    B00: 2
    B01: 2
    B02: 1
    B03: 1
  synthetic_train_per_condition: 5500
  synthetic_val_per_condition: 500
  qa: legacy all-court geometry KP >=4 and renderer-visible KP >=1
  val_trajectory_rule: ceil(10% source train groups), stable SHA-256 ordering with
    seed 42
  renderer_batch_size: 1
  gpu: NVIDIA GeForce RTX 5060 Ti
metrics:
  candidate_unique: 21248
  geometry_rejected: 1314
  qa_accepted: 19934
  qa_rejected: 0
  pending: 0
  failed: 0
  selected_unique_union: 12500
  source_train_groups: 113
  original_train_rows_checked: 6594
  train_val_trajectory_overlap: 0
  physical_curve_duplicates: 0
  video_frames_decoded: 192
  job_wall_seconds: 6332.793268203735
  output_bytes: 2534070855
repro:
  commit: 150eaeb0f25406a437fe3306c011d3ae0e9a0b1f
  branch: codex/court-scene-diversity-data
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. .venv/bin/python knowledge/runs/run-court-scene-diversity-6000-v1/generate.py
    render --output-root /home/kamimura/projects/tennis-lab/data/court_scene_diversity_6000_v1
    && PYTHONPATH=. .venv/bin/python knowledge/runs/run-court-scene-diversity-6000-v1/generate.py
    finalize --output-root /home/kamimura/projects/tennis-lab/data/court_scene_diversity_6000_v1
artifacts:
  run_dir: knowledge/runs/run-court-scene-diversity-6000-v1
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/0-8207086017856490539_2026-10-25T07-39-42-143509460Z_2172746_court-scene-diversity-render-v1.log
  output_dir: data/court_scene_diversity_6000_v1
  plan_audit: knowledge/runs/run-court-scene-diversity-6000-v1/plan-audit.json
  selection_report: knowledge/runs/run-court-scene-diversity-6000-v1/selection-report.json
  subset_audit: knowledge/runs/run-court-scene-diversity-6000-v1/subset-quality-audit.json
  execution_summary: knowledge/runs/run-court-scene-diversity-6000-v1/execution-summary.json
  qualitative: knowledge/runs/run-court-scene-diversity-6000-v1/representative-views.png
  target_binding: knowledge/runs/run-court-scene-diversity-6000-v1/target-binding-interpretation.json
  validation: knowledge/runs/run-court-scene-diversity-6000-v1/validation.txt
parents:
- group-i983-court-dinov3-shared1024-s42
relations:
- to: run-court-novel-view-quality-20261022
  rel: related
papers: []
tags:
- scene-diversity
- fixed-image-budget
- legacy-train-trajectories
- local-gpu
- seed-42
---

#983の公開入力snapshotに記録されたB00〜B03のv3 Court storeから、既存train軌道113本だけを固定して新しい視点をローカルGPUで描画した。各条件6000枚のうちtrain5500枚・val500枚を満たす8個の選択manifestを公開し、共有poolは19,934枚になった。元dataset・3DGS資産は不変で、元validation/test軌道を使用していない。今回の厳密な枚数固定は#1343の全採択production recipeと異なる実験条件である。

| scene | 元train軌道 | 密度level | 候補 | 採択pool | train用pool | val用pool |
|---|---:|---:|---:|---:|---:|---:|
| B00 | 28 | 2 | 7264 | 6935 | 5879 | 1056 |
| B01 | 21 | 2 | 6880 | 6284 | 5641 | 643 |
| B02 | 32 | 1 | 3520 | 3318 | 2856 | 462 |
| B03 | 32 | 1 | 3584 | 3397 | 3119 | 278 |

元metadataのcurve・center・orientation・vertical phase・選択済みviewを直接importし、budget選択やsplit再計画を行わなかった。B01の画角45/67.5/90度も保持した。4096区間の元arc tableとbase countを再現し、旧train全6594行のcamera位置・Kを照合した。追加点のjitterは同じ半径1mの円盤面積一様分布をstable point IDとseed42で決める。旧sample index由来の乱数列は再利用せず、密度変更時の共通ID・camera・教師の不変性をCPUテストした。

描画前にsceneごとの元train軌道のceil10%をstable hash順でval用に予約した。train/valの軌道所属は全条件で固定し、各split内で採択数に比例する軌道スケジュールとarc fractionのmaximin順を使ってnested prefixを選択した。4条件の指定scene別枚数、全条件横断のtrain/val軌道重複0、scene別nested prefix、画像ID・camera・JPEGの一意性を実出力で検査した。条件間の選択unionは12,500枚。画角を無視した物理的同一曲線も全sceneで重複0だった。

旧QAのgeometry all-court KP>=4、renderer visible KP>=1を維持した。21,248候補のうち1314枚は幾何で棄却し、描画19,934枚はすべて採択した。ROI穴率は全採択画像で0だった。共有poolは保存契約上trainのままで、subset manifestのsplitを学習側がtrain/valへ明示的にremapする。公開前に全JPEG・疎教師・QA ledgerを既存strict readerで検証し、生成job内のfinalizeでも再読込してから8manifestを確定した。

| 条件 | trainのtarget KP=0 / <4 | valのtarget KP=0 / <4 |
|---|---:|---:|
| scene-1 | 0 / 164 | 0 / 16 |
| scene-2 | 21 / 220 | 0 / 8 |
| scene-3 | 12 / 143 | 0 / 6 |
| scene-4 | 9 / 99 | 0 / 5 |

B01 poolにはtarget KP0の38枚があり、他courtの可視点で旧QAを通る。上表とscene別のfull/near_full/partial/sparse/none分布は[subset監査](../../runs/run-court-scene-diversity-6000-v1/subset-quality-audit.json)に残した。採択条件と固定manifestを維持して集計しており、生成後に品質閾値を変更していない。合成valの写り方・支持点数と実画像valの違いを、下流評価の解釈で確認する必要がある。

32枚のsmoke、全sceneのcontact sheet、連続軌道動画の代表frameを実見した。B01/B02にはぼけやネット・地面のにじみ、B03には暗い描画が残る。4動画は48frameずつ全192frameのdecodeに成功した。代表B00/B01の奥側KP集中を旧画像・target binding・投影と照合し、同じtargetの近側点がカメラ後方や画面外に出る構成を確認した。[代表画像](../../runs/run-court-scene-diversity-6000-v1/representative-views.png)と[照合値](../../runs/run-court-scene-diversity-6000-v1/target-binding-interpretation.json)を保存した。独立3D GTとの残留alignment誤差は未測定である。

生成と全量検証・finalizeを含むqueue jobは16:39:42〜18:25:14 JST、6332.8秒で成功した。出力rootは約2.36GiB。CPUテスト34件とruff/mypyが合格した。生成処理のためTensorBoard学習曲線は該当しない。学習・実画像評価は親の準備作業へ接続し、scene数による汎化の結論は学習後に判断する。再現手順と8manifestの契約は[生成bundle](../../runs/run-court-scene-diversity-6000-v1/README.md)を参照する。

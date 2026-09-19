---
id: run-slcs-meiji-v9-repair-batch-v1
type: run
title: 'Meiji v9追加2clipをCPU再生成して記録不一致を解消'
provider: codex
date: '2026-09-19'
status: done
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
config:
  data: slcs/meiji_rgb_v9_repair_batch_v1
  clips: [video_001/clip_010, video_001/clip_011]
  stage: infer
  device: cpu
metrics:
  audited_clips: 2
  audit_error_clips: 0
  clip010_ball_positive_weight_fraction: 0.8192307692307692
  clip011_ball_positive_weight_fraction: 0.88268156424581
  clip010_raw_ball_reprojection_mean_px: 16.184548588929264
  clip010_refined_ball_reprojection_mean_px: 7.999695410659509
  clip011_raw_ball_reprojection_mean_px: 14.964027693889701
  clip011_refined_ball_reprojection_mean_px: 7.679828110176651
repro:
  commit: d499ba31
  branch: codex/slcs-real-rgb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    .venv/bin/python -m src.tennis_scene.scripts.build_slcs_dataset device=cpu stage=infer
    'dataset_clip_ids=[video_001/clip_010,video_001/clip_011]'
    dataset_output_directory=slcs/meiji_rgb_v9_repair_batch_v1
    output_dir=tennis_scene/generate/meiji_rgb_v9_repair_batch/s42-001
    paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v9-repair-batch-v1
  output_dir: outputs/tennis_scene/generate/meiji_rgb_v9_repair_batch/s42-001
parents: [run-slcs-meiji-v9-partial-qc-v2]
relations: [{to: run-slcs-meiji-v9-clip009-repair-v1, rel: confirms}]
tags: [slcs, meiji, quality, provenance, cpu, repair]
---

## 考察 / Findings

### 要約

追加の2clipを同じ観測・固定重みでCPU再生成し、公開前チェックと媒体・RGB特徴を含む品質監査が成功。
全体GPU生成の完了を待つ必要がない独立作業なので先行して修復した。旧raw/annotationは退避し、削除しない。

### アーキテクチャ詳細

先行clip009と同じ手順。変更はdeviceと一時dataset/runの出力先だけで、既存DINO特徴をコピーして監査した。
新規学習・metadataの手修正はない。監査済みannotationと生成成果物を本体へstaging renameで差し替え、
copy元との全ファイル一致を確認した。CPU recipeとpromotion記録を差し替え先にも残した。

### メトリクスの解釈

同じ最終正weight maskでball再投影平均はclip010が16.1845→7.9997px、clip011が14.9640→7.6798px。
正weight率は81.92%、88.27%。player支持率はそれぞれ[100%,98.46%]、[100%,83.80%]。
擬似教師の観測整合性であり、独立3D精度ではない。学習がないため収束曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

指定pin・producer・rawの記録が一致する再生成物のみを採用した。不一致原因の断定やPLCS/BLCSの
例外許可は行っていない。旧結果と数値の優劣を比較してモデル改善と称することもしない。

### 既存実験との比較

部分監査2での2件のerrorを解消した。先行clip009と合わせ3clipを退避付きで修復した。
このrunは2clip subset監査であり、継続中の全体56clipの監査成功を意味しない。

### 次に有効な実験

生成完了後に本体の全体監査を行い、もし新たな不整合があれば同じ明示手順で修復する。

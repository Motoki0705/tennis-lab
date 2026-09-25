---
id: run-scene-component-meiji-fullclip-r2-20260925
type: run
task: tennis_scene
sequence: 15
recorded_at: '2026-09-25'
title: 外部ball取り込み後に既定Courtモデルの領域探索が停止
issue: 915
provider: codex
session: 01a0d0f9-6057-7961-ad6b-adc46744d619
date: '2026-09-25'
status: failed
config:
  pipeline: declared_components_v1
  clip: video_000/clip_000
  frames: 1010
  cameras: 3
  ball_source: external_annotation_load
  side_source: confirmed_load
  court_checkpoint_sha256: dd3a396841097e60ff1bc0eabcf7b911e97685e251bf8cc441c100b17276e816
metrics:
  imported_ball_observed_cam0: 841
  imported_ball_observed_cam1: 985
  imported_ball_observed_cam2: 941
  accepted_court_regions: 0
repro:
  commit: b0620d701fd38ffd1d5c6cdbeeb8c0f37744cbc4
  branch: codex/clip-component-store
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
    PYTHONPATH=.:/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T170847Z/dino_extension/lib
    .venv/bin/python tests/benchmarks/component_pipeline.py --repo /home/kamimura/projects/tennis-lab
    --clip /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
    --report /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/clip_components_meiji_20260925
    --device cuda
artifacts:
  run_dir: knowledge/runs/run-scene-component-meiji-fullclip-r2-20260925
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790328158950728526_293737_scene-component-meiji-fullclip-r2-20260925.log
  evaluation: knowledge/runs/run-scene-component-meiji-fullclip-r2-20260925/evaluation.json
parents:
- run-scene-component-meiji-fullclip-20260925
relations: []
papers: []
tags:
- declared_components
- real_clip
- qualification
- failed
---

## 結果と次の判断

外部ball注釈を新BallDetectionOutputのschemaへ取り込み、3camera×1010frameの成果物をclip storeへ保存した。動画SHA・FPS・解像度・0始まりの全frame対応を検査した。観測点はcam0=841、cam1=985、cam2=941。補間・遮蔽推定点も区分を残すが、幾何の実観測maskには含めない。数値confidenceは観測1/非観測0の採用weightで、未校正image_scoreを検出確率へ変換していない。原注釈はモデル支援レビューであり、独立した人手GTとは扱わない。

その後のcourt_detection/cam0で、初期frameの13領域すべてがhybrid幾何条件を満たさず停止した。主な状態はno_jointly_supported_candidate、joint_optimization_failed、insufficient_line_support。Court全frame、人物検出、Re-ID、side確認、GVHMR、統合exportには到達しておらず、pipeline完走・実動画精度の証拠ではない。

今回の既定Court checkpointはSHA dd3a…で、以前このclipで成立した重みとの照合が必要になった。保存されたevaluation.jsonとpipeline_config.yamlをbundleへコピーし、後の再試行による上書きから切り離した。次は以前の成功条件を明示的に再現し、Courtを動画から再実行する。外部ball以外の過去予測へ暗黙に切り替えない。

推論qualificationの失敗であり、学習・TensorBoard曲線はない。

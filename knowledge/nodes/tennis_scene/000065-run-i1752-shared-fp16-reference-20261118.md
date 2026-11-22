---
id: run-i1752-shared-fp16-reference-20261118
type: run
task: tennis_scene
sequence: 65
recorded_at: '2026-11-18'
title: 'fp16単一入口比較: scene一致とBall補助配列の許容超過'
issue: 1752
provider: codex
session: 01a1e957-c745-72f3-b662-6dcb032b4374
date: '2026-11-18'
status: failed
config:
  clip_id: video_000/clip_027
  frames_per_camera: 235
  cameras: 4
  pipeline: pipeline_x3d_dpt
  players_per_side: 2
  court_mode: video_shared
  court_side_mode: video_shared
  shared_action: load
  ball_precision: bf16
  dino_precision: fp16
  vitpose_precision: fp16
  seed: 42
  global_python_profiler: false
metrics:
  cli_complete: true
  strict_comparison: false
  scene_all_arrays_exact: true
  all_discrete_exact: true
  components_compared: 38
  arrays_compared: 378
  different_auxiliary_arrays: 9
  cli_wall_seconds: 655.5828025520022
repro:
  commit: 6bf5b7e132ddfe10a8914c7e75719da6f044f9da
  branch: codex/video-batch-cli
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1 TENNIS_LAB_DINO_EXTENSION=/home/kamimura/projects/tennis-lab/.claude/worktrees/video-batch-cli/outputs/tennis_scene/evaluate/video-batch-runtime-20261117/dino-extension-a2/lib/MultiScaleDeformableAttention.so
    PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/video-batch-cli/outputs/tennis_scene/evaluate/video-batch-runtime-20261117/dino-extension-a2/lib
    .venv/bin/python -B -u knowledge/runs/run-i1752-shared-fp16-20261118/single_reference.py
    run --root /home/kamimura/projects/tennis-lab/.claude/worktrees/video-batch-cli/outputs/tennis_scene/evaluate/video-batch-shared-fp16-20261118/reference
    --expect-commit 6bf5b7e132ddfe10a8914c7e75719da6f044f9da
artifacts:
  run_dir: knowledge/runs/run-i1752-shared-fp16-reference-20261118
  log: knowledge/runs/run-i1752-shared-fp16-reference-20261118/single.log
parents:
- run-i1752-shared-fp16-20261118
relations: []
papers: []
tags:
- video_batch
- gpu_acceptance
- fp16
- numerical_comparison
summary_impact: updated
---

親が以前に許可した1clipだけの比較として、3clip一括の直後の共有queue all jobで027を単一generate_dataset CLIから実行した。同じ固定media、canonical path、共有court/side pin、ダブルス2、DINO/ViTPose fp16、Ball BF16、seed42を使い、component storeとoutput rootを分けた。単一CLIのcompose helperにはpipeline_x3d_dptを明示する計測adapterを使う。forward・batch・schedulerは変更していない。両測定のsrc git treeは同一で、Ballのidentity/dependencies/execution_keyも全4cameraで一致する。

GPU CLIはexit0、scene status=ok、GVHMR配置・scene公開まで完走。7resourceは各1load/unload。CPU AFLinkは4load・明示unload0で別に記録した。CUDA/NVML/heartbeat異常なし。GPU比較は1回で、失敗した比較を再実行していない。

**実行成功と数値比較の合否を分ける。既存の全component float規則（atol1e-6、rtol1e-5、離散値exact）では不合格。** 38componentと最終sceneの計378配列のうち、Ball evidenceの9配列が超過した。34の他componentはすべてexact、Ballの主uv/confidence/observed/point_kind/frame_indicesもexact。全離散値・mask・ID、全JSON payload、scene metadata、最終sceneの29配列もexactだった。

| 差のある補助field | 範囲 | 最大絶対差 |
|---|---|---:|
| evidence.heatmaps | 全4cameraの4配列、計1264要素 | 0.00159219 |
| evidence.patches | 全4cameraの4配列、計10要素 | 0.000964351 |
| evidence.candidate_uv_px | cam3、frame69、候補index3、yの1要素 | 1.005325px |

補助候補のyは単一222.028183px、一括223.033508px。両方candidate_valid=true、score=0.064653486である。候補のscore/cell/valid、選択window/time indexはexact。主検出と最終3D sceneへの差はなかった。低scoreの補助候補であっても、保存evidenceの差を除外して厳密比較を成功扱いにはしない。

[厳密比較の全結果](../../runs/run-i1752-shared-fp16-20261118/comparison-completed.json)、[field名・frame・候補の監査](../../runs/run-i1752-shared-fp16-20261118/difference-audit.json)、[sourceとBall identityの照合](../../runs/run-i1752-shared-fp16-20261118/source-comparison.json)を保存した。C3のBF16物理量基準をC4へ黙って適用せず、許容値も変更していない。原因は未確定で、状態を再利用したBall BF16推論の数値再現性を切り分ける追加GPU実験は未実施。AGENTSの静かなフォールバック禁止は出力決定性の保証ではないが、この比較の不合格と補助evidenceの採否は親に返す。

GPU runtimeは成功している一方、実験の厳密比較は不合格なのでnode status=failed。summaryでは「3clip全段完了、最終scene一致」と「全component厳密一致は未達」の両方へ判断を更新する。69clipは未投入。全3clipの単一逐次時間比較はしておらず、wall比からモデル再利用やfp16単独の高速化率を主張しない。

単一CLI wall 655.583秒、recorder wall 655.431秒。資源・モデルの原値は[集計](../../runs/run-i1752-shared-fp16-reference-20261118/resources-and-models.json)に保存した。

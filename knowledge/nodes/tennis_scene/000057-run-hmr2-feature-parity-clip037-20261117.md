---
id: run-hmr2-feature-parity-clip037-20261117
type: run
task: tennis_scene
sequence: 57
recorded_at: '2026-11-17'
title: 短いダブルスclipでHMR2特徴分離前後を比較する
issue: 1754
provider: codex
status: done
summary_impact: none
config:
  players_per_side: 2
  qualification_main_commit: 18a2fea05291eeaccb6d1dbfc81160e26c4080dd
  base_commit: ab638fb2ab825e84275863552da3f47473e390ed
  clip_id: video_000/clip_037
  seed_per_body_path: 42
metrics:
  main_cli_completed: true
  main_valid_placed_person_frames: 1156
  selected_persons: 4
  segments: 4
  selected_frames_per_person: 145
  gpu_body_comparison_completed: true
  separated_valid_placed_person_frames: 1156
  main_vs_base_max_abs: 0.0
  base_vs_separated_max_abs: 0.0
  base_float_fields_compared: 43
  separated_float_fields_compared: 43
  hmr2_loads_per_path: 1
  gvhmr_loads_per_path: 1
  body_geometry_loads_per_path: 1
  integrated_cpu_passed: 1215
  integrated_cpu_wall_seconds: 50.44
  final_cpu_passed: 1216
  final_cpu_wall_seconds: 50.18
artifacts:
  run_dir: knowledge/runs/run-hmr2-feature-parity-clip037-20261117
  output_dir: outputs/hmr2_feature_acceptance_20261117
  qualification_log: knowledge/runs/run-hmr2-feature-parity-clip037-20261117/qualification-main/queue.log
  qualification_preflight: knowledge/runs/run-hmr2-feature-parity-clip037-20261117/qualification-main/body-preflight.json
  receipt: knowledge/runs/run-hmr2-feature-parity-clip037-20261117/comparison/receipt.json
  base_recorder: knowledge/runs/run-hmr2-feature-parity-clip037-20261117/comparison/base_recorder.json
  separated_recorder: knowledge/runs/run-hmr2-feature-parity-clip037-20261117/comparison/separated_recorder.json
  comparison_log: knowledge/runs/run-hmr2-feature-parity-clip037-20261117/comparison/queue.log
  integrated_source_check: knowledge/runs/run-hmr2-feature-parity-clip037-20261117/integrated-source-check.json
  final_base_check: knowledge/runs/run-hmr2-feature-parity-clip037-20261117/final-base-check.json
  final_validation: knowledge/runs/run-hmr2-feature-parity-clip037-20261117/final-validation.json
parents: []
relations: []
papers: []
tags: []
session: 01a1e957-e495-7ef2-81cd-bf26c4ae465a
date: '2026-11-18'
repro:
  commit: 5a0f48053e6a479ff345cc858002752cd4f1794f
  command: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
    MKL_NUM_THREADS=4 .venv/bin/python -B knowledge/runs/run-hmr2-feature-parity-clip037-20261117/compare.py
    --main-store /home/kamimura/projects/tennis-lab/.worktree-outputs/hmr2-parity-main-20261117/hmr2_qualification/clip_037/store/bfa625f39ba2d39336cc
    --main-config outputs/hmr2_feature_acceptance_20261117/qualification-037.yaml
    --output outputs/hmr2_feature_acceptance_20261117/comparison/clip_037
---

親の継続指示（#1754、worker turn 76a9f971355c49ae855fdac0d08f2ecd）により、clip_035を受入入力から外して短い入力を選び直す。既存の失敗は000056へ保持する。今回はclip_037（292frame、59.94fps、4camera、1920×1080）を最初に試す。受入結果は後述の通り成功した。

mainの既存clip_037計測にも、identity 1と0が同じ290frame候補となる人物対応停止があった。source frame 0を目視すると対象コートはダブルスで、近側2人・遠側2人がいる。既定の1人/側から、既に対応済みの試合形式設定 `player_association.players_per_side=2` を明示し、main単一CLI・base・分離後の全経路を同じ設定にする。人物IDの手動統合や採用閾値の変更は行わない。

mainは別の専用worktreeへorigin/main `18a2fea05` を固定し、source無変更の公開 `src.tennis_scene.scripts.run_pipeline` を呼ぶ。clipの長さは既定frame_sampling上限内なので上限overrideは不要。元datasetは読み取りだけとし、store/exportは専用outputsへ保存する。動画4本のSHAとffprobeによる実frame数・解像度はinput-manifest.jsonへ固定する。

GPUは共有training queueのall枠で1clipずつ処理する。最初の候補が身体段まで到達すれば候補探索を終え、その保存済みbody_view_selectionと幾何/2D/三角測量を固定してbase `codex/scene-inference-runtime`（ab638fb2a）と分離後を比較する。停止した候補のnode・reason・原ログは残す。順序と事前許容差はqualification-plan.jsonを正本とする。

許容差は前回から維持し、rad/betas atol=1e-5・rtol=1e-5、位置/translation/mesh(m) atol=1e-4・rtol=1e-5、再投影(px) atol=1e-3・rtol=1e-5。ID/frame/mask/拒否コードは厳密一致を要求し、非空の有効身体配置を条件にする。各身体経路のseedは42、float32、既定batch上限、allocator cap 10GiBで固定する。

この記録はC6の身体推論境界の受入であり、速度benchmarkや全69clipの受入には広げない。標準pipelineのcheckpoint・再構成方式は変えず、現在のsummaryの判断も変えないためsummary_impactはnone。独立validatorは指定/試行/完了0/0/0。

2026-11-18の再開時、前回のqualification jobは未実行のcancelledとして照合した。CUDA異常で人間がWindowsを再起動し、再開後のbounded nvidia-smiはRTX 5060 Ti / driver 621.24 / 使用747MiBを返した。身体受入の成功には数えない。再投入では、開始前・60秒ごと・終了時のNVML health probeを10秒timeout付きで行い、再び無応答/エラーになれば時刻・処理・エラー全文・Python全thread stackを保存して当該queue workloadを終了する。CUDA例外が出た場合も再試行を繰り返さず親へ返す。これはモデル進捗のポーリングではなく、再起動後に要求されたGPU異常検知である。

2026-11-18再起動後のmain単一CLIはqueue done/exit 0で完走した（UTC 16:05:06–16:24:40）。body_view_selectionはperson 0/cam2、1/cam0、2/cam2、3/cam2の4人物、各1segment・145 source frame。body_placementの有効maskは合計1,156人物×frameで非空だった。開始・定期・終了のNVML health probeは全て正常応答し、CUDA異常は記録されていない。mainのsourceと実コマンドは[再現bundle](../../runs/run-hmr2-feature-parity-clip037-20261117/qualification-main/repro/run.json)、原ログとresolved configは同じqualification-mainへ保存した。元datasetは変更していない。

保存済みstoreを専用comparison出力へ凍結し、CPU preflightで全body artifactのschema/version、4動画SHA、mainとの身体段設定一致、C3 snapshotの原文一致を確認した。候補探索はclip_037で終了し、他clipは実行しない。次のbase対分離後比較jobをall枠へ1件だけ登録した。この時点では先行jobのため待機していたため、GPU比較成功には数えなかった。

分離前後の実GPU受入は成功した。comparison jobもqueue done/exit 0で終了し、[receipt](../../runs/run-hmr2-feature-parity-clip037-20261117/comparison/receipt.json)はaccepted=true。main対C3 base、C3 base対分離後の両比較で各43浮動小数field（GVHMR parameterとbody placementの位置/mesh/再投影/その他診断を含む）の最大絶対差は0.0だった。ID・camera・segment/source frame・mask・拒否コードは厳密一致し、分離後の有効配置も1,156人物×frameだった。事前許容差を緩める必要はなかった。4動画SHAは終了後も一致した。

分離後はHMR2 artifactをClipStoreにpublishした後、memory cache無しの別store instanceから再読込してGVHMRへ渡した。[base recorder](../../runs/run-hmr2-feature-parity-clip037-20261117/comparison/base_recorder.json)と[分離後recorder](../../runs/run-hmr2-feature-parity-clip037-20261117/comparison/separated_recorder.json)は、4人物に対しHMR2/GVHMR/body geometryのload・unloadがそれぞれ1回だった。複数clipで回数が増えない契約は既存の1clip/3clip fake計画テストで補う。CUDA例外・NVML timeout・ハングはこの再開後の2jobでは記録されなかった。これは前回のCUDA異常の原因解消を証明するものではない。

この入力では4人物とも1segment・同じ長さのため、実データだけでは非対称segmentの取り違えを十分検出できない。非対称unit/fake integrationおよび特徴row反転の故障注入がその契約を補う。全69clip・clipをまたぐbatch化・速度改善は今回の受入に含めない。body parameter/mesh等の大きい配列はoutputsの比較directoryに保持し、Gitには小さい結果/再現情報だけを残す。標準checkpointや再構成方式は変わらず、summary_impactはnoneのままとする。

比較完了後に最新C3 `8ef978146`（mainのvideo court変更を含む）を統合した。競合はdefinitionのasset helperとshared court helperの両方を残して解消し、[内容hash検査](../../runs/run-hmr2-feature-parity-clip037-20261117/integrated-source-check.json)でGPU受入時の身体段8fileおよびモデル実装が変わらないことを確認した。統合後のCPU unit/integrationはCUDA_VISIBLE_DEVICES空で1,215 passed / 50.44秒。durations=0の中心caseの最長callは0.99秒、表示setupは最大0.01秒だった。既存の全gallery描画caseは6.48秒で5秒目安を超えるが、000052に記録した固有のstore不変/登録schema描画契約を保ったもので、今回新しく重いcaseを追加した結果ではない。実測の小さい要約は[統合検証](../../runs/run-hmr2-feature-parity-clip037-20261117/integrated-validation.json)、全ログはoutputsに保持する。

初回統合HEAD 6c8d78edcのCIはPython/CPU全10分割/Knowledgeを含め全成功した。その直後にC3 baseが1bef628a9へ進み、knowledge採番とC7のframe_sampling.max_frames既定値2048を取り込む必要が生じた。merge先の予約最大51を保持し、本branchの未マージ記録だけを52・53へ採番し直した。run ID・原ログ・reproは変更していない。新baseでも身体段8fileとモデル実装は受入時のhashを維持し、clip_037の標本格子146点はmax_frames=1024/2048で厳密一致した（[追加確認](../../runs/run-hmr2-feature-parity-clip037-20261117/final-base-check.json)）。したがって実GPUを追加実行せず受入結果を保持する。新base統合後のCPUは1,216 passed / 50.18秒、中心caseの最長callは1.05秒、既存gallery全描画は7.04秒だった。最終HEADのCIはPRに対象SHAとrunを記録する。

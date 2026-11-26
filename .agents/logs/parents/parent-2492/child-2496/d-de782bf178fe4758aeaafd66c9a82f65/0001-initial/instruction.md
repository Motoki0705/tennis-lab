# #2496 共通の速度計測harnessと現行人物モデルの速度基準（親 #2492）

担当issue #2496 の本文が仕様の正本。親 #2492 と、後でこのharnessを使う #2487・#2490・#2491 の「推論速度の計測」節も読み、3子の要求（PT32/PT16/TRT16、batch=1必須とbatch掃引、model-onlyとstage wall time、P50/P95、peak VRAM、cold start・engine buildの分離、parity、実行不能の記録、暗黙fallback禁止）をこのharnessが満たせるようにする。

## 親が確認済みの事実（着手時の手がかり。コードで再確認する）

- 既存のTensorRT部品: `src/utils/models/tensorrt_artifacts.py`・`tensorrt_build.py`・`tensorrt_runtime.py`、`src/tennis_scene/pipeline/tensorrt_assets.py`、`src/tennis_scene/scripts/tensorrt.py`、`src/tennis_scene/scripts/qualify_gvhmr_tensorrt.py`、`src/submodules/models/dino/person_detector.py`、`src/submodules/models/hmr2/feature_extractor.py`、`src/submodules/models/gvhmr/onnx_export.py`。
- 既存の計測記録: `knowledge/runs/run-i2297-clipreid-tensorrt`・`run-i2303-vit-tensorrt`・`run-i2304-dino-tensorrt`・`run-i2336-vision-batch-sweep` と対応する `knowledge/nodes/tennis_scene/` のノード。例: DINO 4-scale Swin-L のPT32 median 268 ms（RTX 5060 Ti、入力750×1333）。これらと整合するか確かめる。
- pipeline実行の計測には `src/utils/inference_recording.py`・`inference_timing.py` がある。今回のharnessはモデル単位の比較用なので、重複させずに再利用できる部分は使い、役割の違いをREADMEに書く。
- 代表clip（合意済み）: video_000 の clip_003・clip_057・clip_004・clip_013・clip_040・clip_028。共通ベンチマーク `data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v1` は読み取り専用。HMR2特徴＋GVHMRの入力は各clipの `scene/components/body_view_selection` の成果を使える。
- ローカルGPUは RTX 5060 Ti 16GB。GPUを使う計測はすべてtraining queueへ `add` する（queue workerは親側で稼働中。自分で `start`・`serve` しない）。

## 進め方

1. harnessの設計を先に決め、unit testで集計・OOM記録・fallback禁止の契約を固める。3子がモデルごとのadapter（前処理・forward・後処理を渡すだけ）で使える形にする。
2. 固定の速度入力（全画面frame集合・人物crop集合）を代表6 clipから作り、ID・sha256付きで保存する。枚数は統計の安定と所要時間の釣り合いで決め、理由を記録する。
3. 現行モデル（DINO、ViTPose-H のflipあり・なし、DINO＋ViTPose-H合計、HMR2特徴＋GVHMR合計）×PT32/PT16/TRT16 をqueueで測る。TRT16の構築やparityが成り立たない条件は、そのまま実行不能として理由を残す。
4. 結果をknowledgeに記録する。

## 返してほしいもの

- harnessとテストのPRを早めに作る（draftでよい）。queueへ計測jobを `add` したら、job IDを宣言して終了してよい。
- 最終的に: PR番号と完全HEAD、harnessの使い方（README）、固定速度入力のID・sha256、現行モデルの速度・VRAM・実行可否の表、既存knowledgeとの整合、knowledge記録。
- 命名は機能・方式が伝わる名前にし、issue番号はPR本文・knowledgeの追跡情報に置く。この基準測定だけに使うscriptはknowledgeのrun bundleへ置く。

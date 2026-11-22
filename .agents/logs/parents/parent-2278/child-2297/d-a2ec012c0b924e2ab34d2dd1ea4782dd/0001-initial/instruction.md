# 担当: #2297（親 #2278 Phase 1-A）

Issue #2297 の本文（Acceptance checklist を含む）が正本。親 #2278 の本文の合意・対象外・共通原則も守る。

## 作業場所・基準
- worktree: `.`（branch `codex/trt-env-protocol`、基準 origin/main 4fd1038e4）。メインworktreeは変更しない。
- `.venv`・`data`・`ckpt` はメインへのsymlink、`outputs` は永続出力。共有 `.venv` には何も install しない（`uv pip install` も不可）。
- TensorRT等の専用環境はrepo外の固定パス（推奨 `$HOME/.venvs/tennis-lab-trt`、[local path]）に作る。共有 `.venv` の torch 2.13.0+cu130 と同じ版を使い、作り方・全パッケージ版を再現手順に記録する。
- GPUを使う実行（計測・変換・PTQ・ncu/nsys）はすべて training queue 経由（`.agents/skills/training-queue/SKILL.md`、`TRAINING_QUEUE_DIR=[local path]`、worktreeのcwdからenqueue）。GPUに直接・同時に投げない。GPU推論中に nvidia-smi を短い間隔で照会しない（WSL2が固まる）。VRAM 16GB を超えるとOOMにならずホストが固まるので、batch/shapeは余裕を持たせる。
- 実験コードは `knowledge/runs/<run-id>[local path]` に置き、`src/` は変更しない。knowledge の記録は knowledge-control skill に従う。

## 進め方（作業単位と時間上限）
1. 環境（上限2時間）: 専用環境の作成、`trtexec --version`・TensorRT import・FP8/INT8 builder 対応の確認。
2. ncu/nsys疎通（上限1時間）: 小さなCUDA処理で DRAM bytes・kernel数・stall理由・Tensor Core 利用率のmetricが取れるか確認。`ERR_NVGPUCTRPERM` なら、エラー出力を添えてその時点で親へ報告（最終報告でよい）し、カウンタ不要の作業を続ける。勝手にOS設定は変えない。
3. 固定入力とキャリブレーションclip（上限2時間）: `data/tennis_multiview/processed/meiji_4cam/dataset` から評価clip `clip_000`・`clip_037` と、それらと重ならないキャリブレーションclipを選定。CLIP-ReIDの実入力（前処理後crop）を再現可能に取り出す方式を決める。#2238 の `knowledge/runs/run-person-stage-precision-meiji-20261118/` の固定入力・手順を参考にする。
4. プロトコル文書とharness（上限3時間）: Issue本文の項目を満たす。条件・モデルを明示指定して実行し、変換失敗・未対応opは非ゼロで止める（PyTorch/FP32へ戻さない）。
5. CLIP-ReID 5条件の実測（上限4時間）: 条件0/1/2/3a/3b の全指標を取得し、knowledge run ノードへ記録。変換不可・著しい劣化は根拠付きで記録すれば達成扱い。

全体の上限は14時間。止めてよいのは次の場合だけ: (a) Acceptanceをすべて満たした、(b) 人間の操作が要る（ncu権限など）でかつ残る全作業がそれに依存する、(c) 共有 `.venv`・依存定義を変えないと進めない、(d) 親の合意範囲の判断が要る。それ以外で時間が残っていれば次の単位へ進む。

## 返す成果
- `Closes #2297` を本文に書いた日本語PR（knowledge/runs の harness・再現手順・プロトコル文書・knowledge run ノード）。CIの `Python` 成功まで確認する。
- 最終報告: Acceptanceごとの達成/未達と証拠パス、CLIP-ReID 5条件の要約表（latency中央値[local path] VRAM・DRAM bytes・kernel数・cosine）、未確認事項、後続（ViTPose-H/HMR2、DINO、GVHMR）向けの注意点。

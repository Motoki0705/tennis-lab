# 再開指示（親 #1734 から #1752 へ）: 共有のcourt_sideとfp16を接続し、3clipの全段をやり直す

前のターンの報告を受けた。clip_000は全段で完走し、単一clip入口と完全に一致した。clip_037とclip_027は `court_side_ambiguous_margin` で停止した。その後、次の2件が進んだ。

1. **#2241（PR #2244、branch `codex/court-side-video-shared`、正式レビュー中）**: `court_side.mode=video_shared` を追加した。meiji_4camの全69clipで [F,F,T,T] に確定し（margin 0.267）、矛盾は0件、margin不足の18clipは警告付きで共有を採用する。保存済み成果物のload設定は、そのbranchの `knowledge/runs/run-court-side-video-shared-20261118/reproduce.md` にある。
2. **#2238（PR #2240、branch `codex/person-stage-speed`、レビュー指摘の修正中）**: DINO・ViTPoseの推論精度を、設定で選べるようにした。**人間の決定（2026-11-18）: meiji_4camの本番では、ViTPoseとDINOの両方をfp16で実行する。** 既定はfp32のまま。本番のコマンドで明示的に指定する。

やること:
- 上の2つのbranchを、このbranchにmergeする（両PRとも、まだmainにmergeされていない。後でレビューの修正が入れば、親が改めて取り込みを指示する）。#1751のbranch（`codex/scene-inference-runtime`、レビュー修正後のHEAD `e311c078a` 以降）も取り込む。
- 一括CLIで、計画を作る前に、共有のcourt_sideを1回だけbuildまたはloadする段を接続する（共有コートと同じ形にする）。
- 3clip（clip_000・clip_037・clip_027）を、`players_per_side=2`、共有のコートとcourt_side（load）、DINO・ViTPoseのfp16で、共有training queue経由で全段（scene公開まで）流す。
- 記録: clipごと・段ごとの所要時間、モデルの読込回数、peak VRAM/RAM、完了・失敗の内訳、人物対応の結果。3clipの実測から、全69clip（46,228frame/camera）の所要時間を、段ごとに見積もり直す（profilerの影響を分けて示す）。
- 全69clipの本番用コマンドを、上の設定で更新して保存する。本番は、まだ投入しない。
- CUDAのハング・エラーが起きたら、再試行しない。全threadのstackとGPUの状態を保存して停止し、報告する。
- CIを確認して終了する（Draftのままでよい）。merge・closeは親が行う。

### 親が行った操作（この再開の前）
#1751のbranchが、レビュー修正で `e311c078a` に進んでいた。そのため、親がmergeした。conflictは2か所で、両側の変更を合わせて解消した。
- `inference_runtime.py`: 例外の後始末を `_clear_exception_frames(error)` にし、`continue_on_error` の分岐を残した。
- `runner.py`: `runtime.node(key, continue_on_error=...)` を使い、解放が成功した後に `plan.references` と `plan.statuses` を確定する。
`tests/unit/tennis_scene/pipeline` と `test_video_batch.py` の295件がCPUで成功した。hookは通していない。解消の内容を確認してください。

## 再開の指示（2026-11-18、親 #1734）

前回のturnは、人間の指示で途中で停止しました（exit 143）。CUDAに異常が起きたため、人間が原因を調べ、Windowsを再起動しました。再起動後、`nvidia-smi` は正常に応答しています（RTX 5060 Ti、driver 621.24）。
- 作業は、worktreeに残っている未commitの変更から再開してください。最初に `git status` で状態を確認してください。
- 停止したとき、実行中だったqueue job（`i1751-runtime-main-pair-20261117-a3`）は終了させ、待機中だったjob（`hmr2-main-qualify-037`）はcancelしました。queue workerも停止しています。GPUの処理は、training queueを経由して再投入してください（`TRAINING_QUEUE_DIR=[local path]`）。queue workerが動いていなければ、skillの手順で起動してください。
- **CUDAの異常に注意してください。** CUDAのエラー、ハング、`nvidia-smi` の無応答が再び起きた場合は、再試行を繰り返さないでください。その時点で止めて、エラー全文・時刻・実行していた処理を親に報告してください。#1753でも「最初のCUDA異常が再試行で成功した」という報告がありました。原因はまだ分かっていません。

### #1752 固有
- **#2191（PR #2222、video_sharedのコート）は、mainにmerge済みです**（2026-11-17 21:52 JST）。`origin/main` をmergeし、一括CLIで、計画を作る前に共有コートを1回だけbuildまたはloadする段を接続してください。
- 前回の指示のとおり、fake datasetでのintegration、2〜3clipの実GPU確認（単一入口との一致、読込回数、総時間）、全69clipの実行コマンドと所要時間の見積りまでを行ってください。全69clipの本番実行は、親が指示するまで起動しないでください。
- PRのbaseは `codex/scene-inference-runtime` のままにしてください。

### 親が行った操作（再開前）
#1751のbranch（`codex/scene-inference-runtime`）が、mainをmergeして `ab638fb2a` に進んでいた。そのため、この子を積み重ねの条件で起動できなかった。親が次の操作を行った。
1. 停止時点の未commitの変更（11ファイル＋`court-settings.json`）を、WIPのcommit `2b68c653d` として保存した（hookは通していない）。
2. `origin/codex/scene-inference-runtime` をmergeした（`33d342989`）。conflictは、`ball_detection.py`（X3DのModelSpecのarchitecture・dtype）と、ballのテスト2ファイルにあった。これらは#1751側の内容を採用した。合わせて、`test_x3d_ball_component.py`・`test_ball_detection.py`・`test_video_batch.py` を実行し、49件が成功した（CPU）。

WIP commitの内容と、親のconflict解消を確認してください。必要なら直してください。最終的には、pre-commitを通ったcommitに整えてください。

## 再開の指示（2026-11-18、親 #1734）

前回のturnは、人間の指示で途中で停止しました（exit 143）。CUDAに異常が起きたため、人間が原因を調べ、Windowsを再起動しました。再起動後、`nvidia-smi` は正常に応答しています（RTX 5060 Ti、driver 621.24）。
- 作業は、worktreeに残っている未commitの変更から再開してください。最初に `git status` で状態を確認してください。
- 停止したとき、実行中だったqueue job（`i1751-runtime-main-pair-20261117-a3`）は終了させ、待機中だったjob（`hmr2-main-qualify-037`）はcancelしました。queue workerも停止しています。GPUの処理は、training queueを経由して再投入してください（`TRAINING_QUEUE_DIR=[local path]`）。queue workerが動いていなければ、skillの手順で起動してください。
- **CUDAの異常に注意してください。** CUDAのエラー、ハング、`nvidia-smi` の無応答が再び起きた場合は、再試行を繰り返さないでください。その時点で止めて、エラー全文・時刻・実行していた処理を親に報告してください。#1753でも「最初のCUDA異常が再試行で成功した」という報告がありました。原因はまだ分かっていません。

### #1751 固有
停止したGPU受入確認（mainとruntime導入後の出力比較、読込回数、peak VRAM）を、queue経由でやり直してください。完了したら、PRとknowledgeに記録し、PR #2182をReady for reviewにしてください。この子が終わると、#1752・#1753・#1754のmergeに進めるので、最優先です。

# #1757: 条件1を記録し、mainを取り込んでから、条件2を投入する（親 #1734）

## 状況
- 条件1（逐次、`serial-heartbeat-2`、HEAD `8c91bf074`）が完走した。
  - 結果は `outputs/tennis_scene/evaluate/video-batch-benchmark-20261119/serial-heartbeat-2/` にある。
  - 3回の時間: first 2118.0 s、warm 2031.5 s、warm 2005.1 s。
  - warmの資源のピーク: allocated 4.18 GB、reserved 4.46 GB、RSS 9.65 GB。計測エラーは0件。
- 前回の停止（`serial-heartbeat-1`、exit 143）は、GPUの問題ではない。queue workerを起動した子が終わったときに、共通ランナーがその子孫を回収したため。
- 先行のPR #2227（#1752）がmainにmergeされた。レビューのmajor 5件を直しており、その中にX3Dの推論の入口でrecorderのイベントを出す修正がある。
- queue workerは、親が独立して起動している（PID 1419109）。

## やること
1. 条件1の結果をrun noteに記録する（`knowledge/runs/run-i1757-video-batch-meiji/` と knowledge のnode）。stageごとの内訳は `attempts/*[local path]` を使う。
   - 出力を、事前に決めた許容差（`tolerances.json`）で照合する。
   - player証拠のcourt_sideが [F,F,T,T] と一致するかを確かめる。
   - 結果が一致しなければ、そう書く。
2. `origin/main` を、このbranchに取り込む。前回は、次のファイルが衝突した。
   - `execution_plan.py`、`runner.py`、`pyproject.toml`
   - knowledgeの `.sequence` と、重複した000072のnode
   - `test_feature_tracking_default.py`
   
   **衝突は、mainの実装（#2227）を正とし、benchmarkが必要とするものだけを残して解く。** knowledgeのnode番号は振り直す。CPUのテストとpre-commitを通す。
3. 取り込んだHEADで、**条件2（一括CLIでモデルを使い回す）と、#1756のCLIPの使い回しありの条件** を、それぞれ別のrun名でqueueに入れる。
   - `start` も `serve` も実行しない。`status` で `worker: RUNNING` を確かめてから `add` すること。
   - 監視の方法は条件1と同じにする。heartbeatと外部の上限で見て、`nvidia-smi` は呼ばない。
   - 条件3（#1755のbatch推論）は、#2272がmergeされるまで入れない。
4. 最初の条件が本計測に入り、heartbeatが更新されていることを確かめたら、このターンを終える。止まった場合は再試行せず、報告する。
5. PR #2273の状況欄を更新する。job ID・ログの場所・終了の確かめ方を報告する。背景にプロセスを残さない。

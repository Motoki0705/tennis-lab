# 再開指示（親 #1734 から #1751 へ）

前のターンの停止報告（a4、mainの基準側でDINOの `DeformableTransformer.forward` 中にheartbeatが180秒途絶）を受け取った。親の判断は次のとおり。

- **観察**: 停止はmain側、つまり既存のコードで起きた。#1751の変更による退行ではない。プロセスはSIGTERMで正常に終了した。その後、`nvidia-smi` は応答しており、GPUは757MiB・0%に戻った。直後に共有queueで実行された別のGPUジョブ（HMR2の037・一括CLIのparity）は、DINOの段を通過している。間欠的な停止とみる。
- **指示**: GPU受入（a5）を**1回だけ**再投入する。自分で設定した共有queueの受付停止flagを解除してから投入する。queue workerが停止していれば起動する。
- **停止時の診断を強化する**: 同じheartbeatの途絶が再発したら、SIGTERMの前に、全threadのPython stackを取得する（`faulthandler.dump_traceback_later` または `py-spy dump --native`。使える方でよい）。合わせて `nvidia-smi -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE` を保存する。そのうえで再試行せず停止し、報告する。原因の調査は人間が行う。
- 受入が成功したら、前回の指示どおり、完成出力の比較・読込回数・peak VRAMをknowledgeとPRに残す。CIが成功したら、PR #2182をReadyにして終了する。merge・closeは親が行う。

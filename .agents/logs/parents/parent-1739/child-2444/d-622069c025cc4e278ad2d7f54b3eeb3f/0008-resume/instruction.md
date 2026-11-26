# 親からの続き（担当親 Joseph）

GPUロックファイルのACLをユーザーが修正した。`[local path]`・`.gate`・`.slot-0`・`.slot-1` の4ファイルは、親が開けることを確認済み。PR #2449 のCIは全件成功している（CI監視はネットワークエラーで止まっただけ）。

- 予約した holdout の ViTPose を training queue 経由で実行する。
  - `TRAINING_QUEUE_DIR=[local path]` を使う。
  - workerを起動・serveしない。
- 完了後、固定済みの設定で採点を一度だけ行い、knowledge に記録する。
  - 対象は3 clip全体と、校正補完clipを除く2 clipの両方。
- #2445（PR #2469）の holdout 採点も、同じ ViTPose 出力を待っている。完了したら、#2445 が読む場所とその完了記録を報告に書く。
- 本日、巨大JSONを `jq` で丸ごと読み、WSLがメモリ枯渇で停止した。大きなJSON・npzは、必要な部分だけをPythonで読む。
- 終わったらPRを更新し、PR番号と完全なHEADを宣言する。

# 親からの続き（担当親 Joseph）

`[local path]` の worker を、親がユーザーsystemdの常駐サービス `tennis-lab-local-queue.service` として起動した。
- `training_queue.sh status` は `worker: RUNNING` を返す。
- 古い「running」9件は取消済みにした。
- GPUロック4ファイルの権限も復旧済み。

前回止めたところから続ける。
- 予約した holdout の ViTPose を、このqueueへ `add` で投入する。workerは起動しない。
- 完了したら、固定済みの設定で一度だけ採点する。対象は3 clip全体と、校正補完clipを除く2 clip。
- 結果を knowledge に記録し、PRを更新する。
- #2445 が読む出力と、完了記録の場所を報告する。
- 大きなJSON・npzは、必要な部分だけをPythonで読む。
- 最後にPR番号と完全なHEADを宣言する。

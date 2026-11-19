# WSL shutdown からの再開（親から、attempt 6）

## 起きたこと（親が確認した事実）

- 2026-11-15 09:20 JST頃、WSLが外部からshutdownされ、再起動した。attempt 5 の service（`tennis-ball-arch-desktop-production-20261113-a5.service`）は inactive。`production-FAILED*`・terminal notificationはない（学習の異常ではなく、外部の停止）。
- serial-status: T1後半・T2前後半は completed、C1-dpt-10000 は running（train.jsonl 6,605行）、C2・C3 は queued。
- Driveで検証済みの最新世代は C1 `checkpoint-00006500`（stage-status の receipt・readback とも verified）。
- `[local path]` は再mount済み。D:上のデータセットコピー `tennis-hot/ball-arch-s42-20261111-fixed-v1` は残っている。D: 空き約286GB、E: ext4 空き約3.5TB。ホストRAMは再起動後に合計約39GBに増えている（.wslconfigが変わった可能性）。
- 別の親 Ian（#1990）が、repo rootの `data/`・`outputs/`・`ckpt/` のsymlinkをD:の実ディレクトリへ切り替える作業をしている。worktree `ball-arch-desktop` の `outputs`、E:のmount、`tennis-hot`、`.training_queue/` には#1859の完了まで触れないと合意済み。

## 依頼

`monitoring-handoff.json` の recovery 手順（未使用の attempt 6）で、C1 の検証済み 6,500 から本番を再開する。

- 旧writer（a5のservice・queue job・GPUプロセス）が止まっていることを確かめてから再開する。queueの a5 job が running のまま残っていれば、その扱いを記録してから進める。
- データ（D:コピー）・env・native build・compile cache・入力を、再起動後のパスで照合する（前回と同じ規則。データの全件SHAはサイズとmtimeが前回照合時から変わっていなければ、前回の照合記録を参照してよい。変わっていれば全件照合する）。
- コード・ガード・recipeは変えない（実行commit `7d7a2e4fa` のまま）。変更が必要になった場合は止めて報告する。
- 復旧の規則は前回と同じ（未検証の状態を採用しない、idempotency keyを再利用しない、旧attemptの証拠を保持）。

## 終了条件

再開後、compile後の最低10回の更新が finite・連続で、VRAM/RAM/D:[local path]。`monitoring-handoff.json` を新しいunit・queue ID・出力root・terminal eventに更新し、commit・pushする。C1の完了は待たない。定期follow-upは登録しない。作業時間の上限は2時間。

止めてよい条件: 6,500の世代から完全な状態を復元できない、データ・環境の照合に失敗する、コード変更が必要になる。最終報告には、再開点、新しいunit/queue/出力root、照合の方法と結果、update秒、現在のRAM・D:・E:の空きを書く。

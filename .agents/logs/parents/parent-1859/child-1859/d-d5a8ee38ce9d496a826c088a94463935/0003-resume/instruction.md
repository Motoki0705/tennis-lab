# 本番a2の停止からの復旧（親から）

## 起きたこと（親が確認した事実）

- 2026-11-13 14:25 JST頃、T1-q06000-q12000 が6,000更新まで学習し、`validation-update-00006000.json` と `.frames.jsonl`、`checkpoint-update-00006000.pt` がローカルにある。
- その6,000の checkpoint 永続化で、Driveからのreadback（`snapshots.restore` の rclone copy）中に guard が発火し、`production-FAILED-T1-encoder_msda.json` が書かれた。host guardの記録は `minimum_available_bytes` 12.39GB（RAMは問題なし）、`maximum_filesystem_growth_bytes` 19,060,776,960 > `storage_stop_threshold_bytes` 19,000,000,000。**原因はストレージguard**（readbackの一時展開が増分を押し上げた）。serial-statusはT1 failed、serviceはまだactive（queueのterminal処理が残っているか確認すること）。
- Windows側の実消費: 起動前にD:の空き約40GB・ext4.vhdx 956.17GBだったのが、現在 D:空き 35,404,603,392 bytes・vhdx 963,181,019,136 bytes。WSL内の増分約19GBのうち、vhdxの実際の増加は約7GBだった（解放済みブロックの再利用）。

## 親の判断

前回の「WSL新規書込み20GB以内」は、D:の物理的な空きを守るための代理の指標だった。実際の消費と合わないので、**guardの容量条件を「D:の実際の空き」に置き換える**。

- 停止条件: `[local path]` の available（statvfs）が **10 GiB 未満** になったら停止する（既存のRAM下限ガードは維持）。WSL全体の増分による停止は外す。増分は記録として残してよい。
- guard の変更はテストを付けて commit・push し、CI `Python` 成功を確認してから、そのcommitで新しいattemptとして再開する（既存のattempt番号・idempotency keyは再利用しない。手順は `recovery.md` / `monitoring-handoff.json` の recovery に従う）。
- 復旧時、T1の6,000をやり直さずに済むかを既存の復旧ツールの規則の範囲で判断する。Driveにuploadされた6,000の世代を独立readbackで完全に検証できるなら、それを使ってよい。検証できなければ、最新の検証済み世代（5,500）から再実行する。どちらを選んだか、その根拠を記録する。規則を緩めて未検証の状態を採用しない。
- #1859の自分の出力のうち、Driveに検証済みのコピーがあり、再開・復旧に不要なローカルファイル（失敗attemptの一時ファイル、probeの残り等）は、記録を残して整理してよい。`desktop-resume-20261113`（約15GB、データセットのhardlinkを除く）の中身を確認し、再開に必要なenv・native build・compile cacheは残す。#1899や他issueの成果物には触れない。

## 終了条件（前回と同じ考え方）

新attemptを systemd user service で切り離して起動し、再開後の最低10回の更新が finite・連続で、VRAM/RAM/D:空きが上限内であることを実ログで確認したら終了する。T1を5,500から再実行する場合も、10更新の確認で終わってよい。`monitoring-handoff.json` を新しいunit名・queue ID・terminal eventのパス・出力rootに更新してcommit・pushする。T1完了やT2の節目は待たない。定期follow-upは登録しない。

止めてよい条件: 6,000/5,500のどちらからも完全な状態を復元できない、guard変更のテストやCIが通らない、D:の空きが10GiBに近く再開できない。最終報告には、選んだ再開点と根拠、新しいunit/queue/出力root、guard変更のcommit・PR、整理したファイルと解放量、現在のD:空きを書く。

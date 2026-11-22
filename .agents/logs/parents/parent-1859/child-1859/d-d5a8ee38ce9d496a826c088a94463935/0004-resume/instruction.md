# 本番a3の停止からの復旧（親から、2回目）

## 起きたこと（親が確認した事実）

- 2026-11-13 16:31 JST頃、T2-q00000-q06000 の更新1,283付近の backward 中に host RAM ガードが発火した（`production-FAILED-T2-encoder_msda.json`、`trigger_reason: host_available_ram`、`minimum_available_bytes` 4,155,387,904 < 4 GiB）。D:の空きは 24.4GB で問題なし。serviceは16:33に failed で終了し、queue workerも終了した。
- 学習の数値は正常だった（1〜1,283更新で連続・有限。凍結解除後は約3.3秒[local path]、allocated 10.8GB）。Driveで検証済みの最新世代は `checkpoint-00001000`。
- 同じ時間帯に、別sessionのworktree（`log-publication-completion`、`standalone-1934`）で pytest-xdist のworkerが十数個（各約0.6〜0.8GB RSS）動いていた。親が並行して走らせた `du` も dentry cache を増やした可能性がある。学習プロセス自身のRSSの推移は記録がなく、外部要因か学習側の増加かを完全には切り分けられていない。

## 親の判断

他のsessionのテストは今後も起きるので、一時的なRAM不足で本番を止めて毎回復旧するのは割に合わない。**RAMガードを「止める」から「更新の間で待つ」に変える**。

- 更新を始める前に host available RAM を確認し、**6 GiB 未満なら次の更新を始めずに待つ**（6 GiB 以上に戻ったら再開する）。待った時刻・長さ・その時の available・学習プロセス（親と DataLoader の子を含む）のRSS合計を記録する。待機中はGPUのメモリを保持したままでよい。待機の上限は設けない（D:・RAMの回復を待つのが目的）。ただし待機が30分を超えたらその旨をログに1行残す。
- 更新の途中でも、**available が 2 GiB 未満になったら止める**（ホストのフリーズ・OOM killerを避ける最後の安全装置）。停止時の挙動は今と同じ（状態保存・次へ進まない）。
- 学習プロセスのRSS合計（親＋子）を各更新の記録に入れ、外部要因と学習側の増加を区別できるようにする。
- D:の空き 10 GiB 未満での停止、allocator 80%、checkpointing範囲、その他のrecipeは変えない。
- 変更はテスト付きで commit・push し、CI `Python` 成功を確認してから、そのcommitで新しいattempt（4）として、Driveで検証済みの最新世代（1,000）から再開する。復旧の規則は前回と同じ（未検証の状態を採用しない、idempotency keyを再利用しない）。

## 終了条件

再開後、compile後の最低10回の更新が finite・連続で、VRAM/RAM/D:が上限内であることを実ログで確認したら終了する。`monitoring-handoff.json` を新しいunit・queue ID・出力root・terminal event・待機ログのパスに更新し、commit・pushする。待機イベントの記録先（ファイルとキー）も handoff に書く（親が監視に使う）。T2の完了は待たない。定期follow-upは登録しない。

止めてよい条件: 1,000の世代から完全な状態を復元できない、テストやCIが通らない、D:の空きが10 GiBに近い。最終報告には、再開点、新しいunit/queue/出力root、guard変更のcommit・PR、待機ログの場所、現在のRAM・D:の空きを書く。

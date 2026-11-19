# 人間の指示による停止からの再開（親から、attempt 7、C3）

## 起きたこと（親が確認した事実）

- 2026-11-15、人間の指示で attempt 6 の service（`tennis-ball-arch-desktop-production-20261113-a6.service`）を親が `systemctl --user stop` で止めた。`production-FAILED-C3-encoder_msda.json` は `external_signal`（学習の異常ではない）。terminal notification とアップロードは記録済み。
- serial-status: T1・T2・C1・C2 は completed（評価・Drive照合済み）。C3 は running 表示のまま 45 更新で停止。C3 の検証済み世代は初期化直後の 0 だけ。
- 2026-11-16、別の親 Ian（#1990）が repo root の `data/`・`outputs/`・`ckpt/` を E: へのsymlinkから D: の実ディレクトリへ切り替えた。
  - worktree `ball-arch-desktop` の `outputs` は、D: の新しい空ディレクトリ（main 側 `.worktree-outputs/ball-arch-desktop`）を指すようになった。
  - E: 上の旧 outputs の実体（`[local path]` の `desktop-recovery-*`、`persistence/`、`env-original`、a4 の `dataset`）は移動も削除もされていない。親が a6 root の `serial-status.json`・`persistence/`・`env-original/bin/python` の存在を確認済み。
  - ホームの `tennis-hot/ball-arch-s42-20261111-fixed-v1` と `.training_queue/` には触れていない。
  - dataset の版は D: の `data/` に置かれなくなったが、本番は hot copy と E: の a4 `dataset` を絶対パスで読むので、`storage_tier hydrate` は不要のはず。必要になったら止めて報告する。
- queue の running には、a5（PID 不在の古い記録）など、PID のない古い記録が残っている。証拠として保持し、触らない。queue worker（serve）は動いている。
- 親確認時点: D: 空き約397GB、E: 空き約3.4TB、ホームRAM available 約31GB。GPU に計算プロセスはない。

## 依頼

`monitoring-handoff.json` の recovery 手順で、a6 root（`desktop-recovery-20261114-a6-nvme`）から attempt 7 を作り、C3 の検証済み状態（0）から本番を再開する。

- NEW_ROOT は handoff の例と同じく E: の data volume 上（`.../arch_efficiency/desktop-recovery-20261116-a7-nvme` などの未使用の新しい root）に作る。bootstrap は E: 入力の hardlink を保持するので、D: には置かない。handoff の `OLD_ROOT` を a6 root にし、`NEW_ROOT` は日付を正しくした未使用名にする。
- 旧writer（a6 の service・queue job・GPU プロセス）が止まっていることを確かめてから再開する。
- データ（D: hot copy）・env・native build・compile cache・入力を照合する（前回と同じ規則。bootstrap の全件 SHA 照合はそのまま行う）。
- `ckpt/` の切替で、本番が参照する入力（初期重みなど）の解決先が変わっていないかを確認する。変わっていて内容が一致しない、または解決できない場合は止めて報告する。
- コード・ガード・recipe は変えない（実行 commit `7d7a2e4fa` のまま）。変更が必要になった場合は止めて報告する。
- 復旧の規則は前回と同じ（未検証の状態を採用しない、idempotency key を再利用しない、旧 attempt の証拠を保持）。

## 終了条件

再開後、compile 後の最低 10 回の更新が finite・連続で、VRAM・RAM・D:・E: が上限内であることを実ログで確認したら終了する。`monitoring-handoff.json` を新しい unit・queue ID・出力 root・terminal event に更新し、commit・push する。C3 の完了は待たない。定期 follow-up は登録しない。作業時間の上限は 2 時間。

止めてよい条件: 0 の世代から完全な状態を復元できない、データ・環境の照合に失敗する、コード変更が必要になる。

最終報告に書くこと:
- 再開点
- 新しい unit・queue ID・出力 root
- 照合の方法と結果
- update 秒
- 現在の RAM・D:・E: の空き

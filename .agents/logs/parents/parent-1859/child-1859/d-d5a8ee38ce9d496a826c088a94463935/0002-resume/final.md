引継ぎ条件を満たしました。本番は自律継続中です。

- branch: `codex/ball-arch-desktop-20261113`
- 実行commit: `cb3243ed4`
- 報告HEAD: `ee59fc0cc`
- [PR #1950](https://github.com/Motoki0705/tennis-lab/pull/1950)、[PR #1955](https://github.com/Motoki0705/tennis-lab/pull/1955) は各HEADのCI Python成功。CPU・CUDA検証、pre-commit、knowledge検証も成功しました。

T1の**5,501〜5,511更新**でloss・勾配・LRが有限、sample位置21,501〜21,511が連続しました。本番のpeak allocated／reservedは10.804／12.595GB、host available最小16.152GBでガード内です。

T1/T2はCNN各stage＋MSDA chunkのcheckpointingに固定しました。

| 隔離probe | peak allocated／reserved GB | 更新秒 |
|---|---:|---:|
| T1：MSDAのみ | 13.534／13.652 | OOM |
| T2：MSDAのみ | 13.509／13.590 | OOM |
| T1：CNN＋MSDA | 10.804／12.595 | 3.77 |
| T2：CNN＋MSDA | 10.804／12.533 | 3.16 |
| C1：checkpointingなし | 9.397／10.475 | 1.81 |

成功probeは各3更新、秒はcompile後2更新の中央値です。C2/C3は未計測です。

本番unitは `tennis-ball-arch-desktop-production-20261113-a2.service`。service PID `1008145`、controller PID `1008325`、T1 PID `1018768`、学習PGID `1008200`です。共有queueはresource=all／runningを確認しました。queue ID、terminal event、節目の判定条件、正確な復旧コマンドは [監視JSON](https://github.com/Motoki0705/tennis-lab/blob/ee59fc0ccbb0fda58158c3d93a2480dad0ce4846/knowledge/runs/run-ball-arch-desktop-20261113/monitoring-handoff.json) と [README](https://github.com/Motoki0705/tennis-lab/blob/ee59fc0ccbb0fda58158c3d93a2480dad0ce4846/knowledge/runs/run-ball-arch-desktop-20261113/README.md) にコミット済みです。

Drive保存先は `outputs/ball_detection/train/arch_efficiency/s42-20261111/desktop-attempts/20261113-a1/`。引継ぎ資料も保存・readback済みです。Eへの全1,917ファイル・19,243,387,824 bytesのコピーは照合済み。Dのfilesystem増分は最終確認時**約16.75GB**です。初回は19GBガードで更新前に停止しましたが、一時download約1.54GBと派生cache約0.31GBを整理し、full stateを照合してattempt 2へ復旧しました。

T1の12k評価、T2の500保存・600→601解除、残条件の完走・評価は親の監視に引き継ぎます。定期follow-up、merge、issue closeは行っていません。
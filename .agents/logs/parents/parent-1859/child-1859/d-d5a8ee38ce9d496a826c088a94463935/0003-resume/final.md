復旧・引継ぎを完了しました。検証済み6,000世代の全11ファイルとfull stateを照合し、T1を追加更新なしで完了。T2は更新1〜11／sample 10,001〜10,011が連続し、compile後10更新のloss・勾配・LRが有限でした。本番は自律継続中です。

branchは `codex/ball-arch-desktop-20261113`。guard実装 `6ddf48cfa`、修正・実行commit `2cb49a907`、報告HEAD `ce5b55b0c`。[PR #1955](https://github.com/Motoki0705/tennis-lab/pull/1955)の[実行commit CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43374423329)・[報告HEAD CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43379246685)はPython成功、Knowledge CIも成功しました。

- guard：D: available **10GiB未満で停止**。RAM下限4GiB・allocator80%を維持。
- T1 12k validation：observed **85,473**、mean **35.3367px**、median **8.9393px**、P95 **176.7366px**、≤20px **74.1907%**。予測を含めDrive保存・readback済み。
- T2確認prefix：allocated／reserved **7.147／9.737GB**、compile後中央値 **1.835秒[local path]、RAM available最小 **17.03GB**。CNN解除後は親の確認対象です。
- checkpointingはCNN各stage＋MSDA chunkを維持。MSDAのみの候補はT1/T2ともOOM。候補別VRAM・秒は[測定記録](https://github.com/Motoki0705/tennis-lab/blob/ce5b55b0c816dcf0283aa6d88b104031051bcf3e/knowledge/runs/run-ball-arch-desktop-20261113/probe-summary.json)にあります。C2/C3のGPU実測は未確認です。
- 旧attemptの重複tensorとreadback cacheを整理し、**1,476,173,824 bytes**のWSL割当を解放。env・native build・compile cacheを保持しました。現在のD: availableは **31.61GB（29.44GiB）**。元実験のEコピー1,917ファイル・19.24GBも照合済みです。

新rootは [desktop-recovery-20261113-a3](https://github.com/Motoki0705/tennis-lab/blob/ce5b55b0c816dcf0283aa6d88b104031051bcf3e/outputs/ball_detection/train/arch_efficiency/desktop-recovery-20261113-a3)。unitは `tennis-ball-arch-desktop-production-20261113-a3.service`、service PID **1408456**、受入時T2 PID **1477580**[local path] **1408505**です。

queue ID：
`0-8205450230444034201_2026-11-13T06-02-49-555965798Z_1408418_ball-arch-desktop-production-a3`

Drive保存先は `outputs/ball_detection/train/arch_efficiency/s42-20261111/desktop-attempts/20261113-a1/`。監視資料57ファイルも独立readback済みです。

terminal event・state/log/receipt・attempt 4の復旧コマンドは[監視handoff](https://github.com/Motoki0705/tennis-lab/blob/ce5b55b0c816dcf0283aa6d88b104031051bcf3e/knowledge/runs/run-ball-arch-desktop-20261113/monitoring-handoff.json)と[README](https://github.com/Motoki0705/tennis-lab/blob/ce5b55b0c816dcf0283aa6d88b104031051bcf3e/knowledge/runs/run-ball-arch-desktop-20261113/README.md)へ記録しました。T2の500保存・600→601解除、残4条件の完走・評価を親へ引き継ぎます。受入後の進捗polling・定期follow-upは行っていません。
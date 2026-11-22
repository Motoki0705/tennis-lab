C1の検証済み6,500からattempt 6を再開しました。6,501〜6,511更新（compile後10回）はfinite・連続で、VRAM/RAM/D:[local path]。本番は自律継続中です。

- branch: `codex/ball-arch-desktop-20261113`
- 実行commit: `7d7a2e4fa`（コード・ガード・recipe変更なし）
- 報告commit: `67c3017b8`[local path] #1955](https://github.com/Motoki0705/tennis-lab/pull/1955)[local path] CI成功](https://github.com/Motoki0705/tennis-lab/actions/runs/43660839895)
- unit: `tennis-ball-arch-desktop-production-20261113-a6.service`
- queue: `0-8205296550702081252_2026-11-15T00-44-09-297918747Z_280792_ball-arch-desktop-production-a6`
- 出力root: `outputs/ball_detection/train/arch_efficiency/desktop-recovery-20261114-a6-nvme/`

D:のデータ2,925件は全サイズ・SHA一致（270.94秒、追加コピー0）。env・native・cache・入力も照合済みです。更新中央値1.817秒、data_wait中央値1.040msでした。

10:10 JST時点の空きはRAM **28.09 GiB**、D: **285.70 GiB**、E: **3,513.06 GiB**です。

[監視・復旧の引継ぎ](https://github.com/Motoki0705/tennis-lab/blob/67c3017b840cb0e94a40a72675de141784cd9186/knowledge/runs/run-ball-arch-desktop-20261113/monitoring-handoff.json)を更新・pushし、証拠はDrive保存・readback済みです。C1〜C3の完走・評価は親の監視へ引き継ぎます。定期follow-upは登録していません。
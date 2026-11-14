**attempt 5をT2の検証済み1,000更新から再開し、compile後10更新の有限値・連続性を確認しました。本番は自律継続中です。**

- Dコピー先: `[local path]`。2,925 payload・157,028,755,547 bytesの全サイズ[local path]。コピー＋照合は**45分31秒**。E正本は変更せず、env・native・cacheも検証済みです。
- 後続10更新の中央値は**update 3.280秒[local path] 1.013ms**。a4参考値は3.453秒[local path]。ただしa4当時もD上だったため、HDD学習との比較値はありません。
- VRAM最大 allocated **10.80GB／reserved 12.76GB**。最終確認のavailableはRAM **13.25GiB**、D **297.77GiB**、E **3,587.68GiB**。各ガードの閾値内です。
- [PR #1955](https://github.com/Motoki0705/tennis-lab/pull/1955)：実行commit `7d7a2e4fa`、報告HEAD `bcb217262`。[実行CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43565812493)・[報告CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43572028500)のPython成功。選択テスト51件、pre-commit成功です。

unitは `tennis-ball-arch-desktop-production-20261113-a5.service`、queue IDは `0-8205359605821449085_2026-11-14T07-13-14-178550914Z_1105362_ball-arch-desktop-production-a5`。出力rootは `outputs/ball_detection/train/arch_efficiency/desktop-recovery-20261114-a5-nvme/` です。

[monitoring-handoff.json](https://github.com/Motoki0705/tennis-lab/blob/bcb21726223caa38f46916a321852e677df1811e/knowledge/runs/run-ball-arch-desktop-20261113/monitoring-handoff.json)にPID/PGID、terminal event、待機・容量ガードのログ、未使用attempt 6での復旧コマンドを記録しました。引継ぎ資料75ファイルはDrive保存・独立readback済みです。

T2以降の完走・評価は親へ引き継ぎます。定期follow-upは登録していません。
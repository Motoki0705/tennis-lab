第1段を [draft PR #2023](https://github.com/Motoki0705/tennis-lab/pull/2023) にまとめました。

- branch: `codex/storage-tiering`
- commit: `4d1e1588fb9941e542e9efb06cbe35afd22148cd`
- [対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43583872667): **Python成功**。全14ジョブ成功、作業ツリーはクリーンです。

実装・設計の要点です。

- dataset版は設定で宣言し、outputsは標準のrun-id境界から発見します。曖昧な旧配置は未管理として報告し、ckptはミラーのみです。
- ミラー時は両側の全件SHA-256、削除前はD:再ハッシュとE:metadata不変を確認します。書込み・使用中はqueue、fd/mmap、静止待ち、前後snapshotで検査します。
- hardlinkは単位内で維持し、外部linkが残る場合の回収見積りと実際の空き容量差を記録します。
- 手動hydrateと、共通入力境界で取得CLIを案内するエラーを追加しました。新worktreeのD: outputs作成CLIも実装しています。

検証は関連 **392件成功**、基盤・出力規約等 **635件成功**（重複あり）。新規57件の最長は0.919秒です。pre-commit、明示ruff/mypy、systemd unit検証も成功しました。CIで判明した既存テストのumask依存はmainでも再現し、fixtureを修正しています。

実機dry-runは199B・114,958Bの2単位に限定し、0.431秒・0.422秒で成功しました。atime・全metadata・既存symlinkのstat値は不変で、manifestや状態ファイルは作成していません。

**第2段前に親判断が必要です。**

- 既存の336時間worktree削除は、D: outputsを照合なしで消すため統合が必要です。推奨はD:永続領域へoutputsを置き、worktreeからD:内symlinkで参照する方式です。
- user権限で読めないプロセスを検査する仕組みと、通知配送先が必要です。
- 初版はpending/running queueがあれば全単位を保持します。常時稼働との併用には参照単位を明示する連携が必要です。
- main outputsを共有する既存aliasは、ミラーとの対応整理が必要です。

移行案は、停止時間調整→E:index→リンク退避・D: root作成→ckpt全体と使用中単位のhydrate→入力・hardlink確認→使い捨て単位で実削除・復元・容量・通知確認→timer有効化です。初期indexは432GiB・80〜150MiB/sの仮定で約49〜92分[local path]、取得時間は別途必要です。

実機の配置切替・実削除・unit導入、merge・issue closeは未実施です。validatorは指定なし、試行／完了0／0です。
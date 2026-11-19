[draft PR #2023](https://github.com/Motoki0705/tennis-lab/pull/2023) を更新しました。

- branch: `codex/storage-tiering`
- HEAD: `eb78ab8b1a897d0d147540c39f88aa0761db85d1`
- [対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43611072890): **Python含む全14ジョブ成功**

queue監視・関連引数・通知helperを削除しました。プロセス検査は同UIDに限定し、他UIDの除外数をログ・監査記録へ残します。outputsはD:`<main>[local path]<name>[local path]`へ永続化し、E:も同じ相対パスにしました。main outputs aliasの扱い、scanのサイズ集計、README 3言語も反映済みです。

検証は **533件成功[local path]。今回追加10件の最長は0.199秒です。hardlink成立、worktree削除後のoutputs保持、他UID除外・同UID検査不能の拒否を確認しました。pre-commit・静的検査も成功しています。

実機では114,958B・7ファイルの小単位を読取専用検証し、exit 0でした。対象metadata・atime・既存symlinkは不変で、状態ファイルも作成していません。

**実機cycle成功は未達です。** 配置切替前のためexit 1となります。他UID36件は除外できていますが、同UIDのsystemd・sshd・gpg-agent等23件が検査不能で、移行後も清掃を止めます。親判断として、呼出元UIDだけを調べる権限付き読取専用helperを推奨します。別案はサービス側の検査権限を調整する方法です。

未分類E:dataの割当量です。payloadは読まず、symlink先を含めていません。

| 領域 | 割当量 |
|---|---:|
| ACCAD | 0.2405 GiB |
| smplh | 0.2438 GiB |
| smplx | 0.1647 GiB |
| samples | 0.1851 GiB |
| synthetic_data_generation | 18.7052 GiB |
| tennis_multiview | 74.2465 GiB |
| tennis_multiview_linux_retained | 72 KiB |

第2段には、旧E:の独立outputsを永続パスへrenameして再indexする手順を追加しました。その後、ckpt全体・必要単位をD:へ取得し、各worktreeを永続領域またはmain outputsへリンクします。検査権限の解決後、実削除・復元・容量確認を経てtimerを導入します。index見積りは従前の仮定で49〜92分[local path]、hydrateは別途です。

配置切替・実削除・unit導入・merge・issue closeは未実施です。validatorは指定なし、試行／完了0／0です。